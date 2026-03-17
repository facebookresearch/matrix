# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
import asyncio
import glob
import logging
import os
import random
import time
import traceback
from collections import defaultdict
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Type

import hydra
import numpy as np
import ray
import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from matrix import Cli
from matrix.utils.ray import get_ray_address

from .agent_actor import AgentActor, ContainerExecutionAgent, LLMAgentActor
from .agent_utils import (
    HistPair,
    RayDict,
    get_ray_actor_class,
    setup_logging,
)
from .dataset_loader import BaseDatasetLoader
from .orchestrator import (
    BaseResourceClient,
    DeadOrchestrator,
    Orchestrator,
    SequentialOrchestrator,
)
from .dispatcher import Dispatcher
from .sink import Sink

# Re-export all public names for backward compatibility
__all__ = [
    "RayDict",
    "HistPair",
    "Orchestrator",
    "SequentialOrchestrator",
    "DeadOrchestrator",
    "BaseResourceClient",
    "BaseDatasetLoader",
    "BaseMetricsAccumulator",
    "AgentActor",
    "ContainerExecutionAgent",
    "LLMAgentActor",
    "Sink",
    "Dispatcher",
    "ScalableTeamManager",
    "P2PAgentFramework",
    "main",
]

logger = logging.getLogger(__name__)


# ==== Configurable Metrics Accumulator ====
class BaseMetricsAccumulator(abc.ABC):
    def __init__(self):
        self.overall_metrics = defaultdict(list)

    @abc.abstractmethod
    def accumulate(self, orchestrator: Orchestrator):
        pass

    def done(self):
        result = {}
        for metric, value in self.overall_metrics.items():
            if isinstance(value, list) and len(value) > 0:
                result[metric] = sum(value) / len(value)
            else:
                result[metric] = value
        return result


class ScalableTeamManager:
    """Manages teams with multiple actors per role using Dispatchers for routing and load balancing"""

    def __init__(self, simulation_id: str, dispatcher_config: Optional[DictConfig] = None):
        self.simulation_id = simulation_id
        self.teamConfig: Dict[str, Tuple[Type, DictConfig]] = {}
        # Sink actor handle - must be created first
        self.sink: Optional[ray.actor.ActorHandle] = None
        # Dispatcher per role (excluding sink)
        self.dispatchers: Dict[str, ray.actor.ActorHandle] = {}
        # Agent handles per role (must be kept alive to prevent GC)
        self.agents: Dict[str, List[ray.actor.ActorHandle]] = {}
        # Dispatcher ray resource config
        self.dispatcher_ray_resources: dict[str, Any] = {}
        self.dispatcher_debug: bool = False
        if dispatcher_config:
            if "ray_resources" in dispatcher_config:
                self.dispatcher_ray_resources = OmegaConf.to_container(
                    dispatcher_config["ray_resources"], resolve=True
                )
            self.dispatcher_debug = dispatcher_config.get("debug", False)

    def create_role(self, role_name: str, agent_config: DictConfig, resources):
        """Create agents for a role. _sink must be created first."""
        is_sink = role_name == "_sink"

        if not is_sink and self.sink is None:
            raise ValueError("Sink (_sink) must be created first")

        count = 1 if is_sink else agent_config.get("num_instances", 1)
        ray_resources: dict[str, Any] = {}
        if "ray_resources" in agent_config:
            ray_resources = OmegaConf.to_container(  # type: ignore[assignment]
                agent_config["ray_resources"], resolve=True
            )

        agent_class = get_ray_actor_class(agent_config._target_)

        # Sink should not restart; other actors restart infinitely
        max_restarts = 0 if is_sink else -1

        # Create Dispatcher for non-sink roles
        dispatcher = None
        if not is_sink:
            DispatcherActor = ray.remote(Dispatcher)
            dispatcher = DispatcherActor.options(
                name=f"dispatcher_{role_name}",
                namespace=self.simulation_id,
                max_restarts=0,
                **self.dispatcher_ray_resources,
            ).remote(role=role_name, debug=self.dispatcher_debug, sink=self.sink, namespace=self.simulation_id)
            self.dispatchers[role_name] = dispatcher
            logger.info(f"Created dispatcher for role: {role_name}")

        agents = []
        for i in range(count):
            ray_name = f"{role_name}_{i}"
            kwargs = {
                "id": f"{self.simulation_id}_{role_name}_{i}",
                "agent_id": role_name,
                "config": agent_config,
                "resources": resources,
            }
            if not is_sink:
                kwargs["sink"] = self.sink
                kwargs["dispatcher_name"] = f"dispatcher_{role_name}"
                kwargs["namespace"] = self.simulation_id
                kwargs["ray_name"] = ray_name

            agent = agent_class.options(
                name=ray_name,
                namespace=self.simulation_id,
                max_restarts=max_restarts,
                **ray_resources,
            ).remote(**kwargs)

            logger.info(
                f"Created agent: {role_name} id={agent._actor_id.hex()} max_restarts={max_restarts}"
            )
            agents.append(agent)

        if is_sink:
            self.sink = agents[0]

        self.agents[role_name] = agents

        self.teamConfig[role_name] = (
            agent_class.__ray_metadata__.modified_class,
            agent_config,
        )

        return agents

    async def initialize_team(self, team: Dict[str, List[ray.actor.ActorHandle]]):
        """Initialize all agents and wire Dispatchers.

        Args:
            team: Dict mapping role -> list of actor handles (from create_role return values)
        """
        # Health-check all actors and dispatchers
        all_actors = [self.sink]
        for role_handles in team.values():
            all_actors.extend(role_handles)
        all_actors.extend(self.dispatchers.values())

        logger.info(f"Checking Ray actor health for {len(all_actors)} actors (including dispatchers)")
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    *[handle.check_health.remote() for handle in all_actors if handle is not None]  # type: ignore[union-attr]
                ),
                timeout=10 * len(all_actors),
            )
        except Exception as e:
            logger.error(
                f"Failed to start Ray actors, check cluster resource utilization. {repr(e)}"
            )
            raise e
        logger.info("Checking Ray actor health done...")

        # Wire Dispatchers to each other
        for role_name, dispatcher in self.dispatchers.items():
            await dispatcher.set_dispatchers.remote(self.dispatchers)

        logger.info(f"Dispatchers wired: {list(self.dispatchers.keys())}")

    def get_team_config(self):
        """Get team config dictionary for orchestrator routing"""
        return self.teamConfig

    async def shutdown(self):
        """Shutdown Dispatchers (which send sentinels to agents), then Sink."""
        # Shutdown dispatchers first (sends sentinels to agents)
        for role_name, dispatcher in self.dispatchers.items():
            try:
                await dispatcher.shutdown.remote()
            except Exception as e:
                logger.warning(f"Error shutting down dispatcher {role_name}: {repr(e)}")

        # Then shutdown sink
        if self.sink:
            try:
                await self.sink.shutdown.remote()
            except Exception as e:
                logger.warning(f"Error shutting down sink: {repr(e)}")


class P2PAgentFramework:
    def __init__(self, sim_index: int, cfg: DictConfig):
        self.sim_index = sim_index
        self.simulation_id = (
            cfg.get("simulation_id", datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
            + f"_{sim_index}"
        )
        self.cfg = cfg
        self.data_loader: BaseDatasetLoader = None  # type: ignore[assignment]

        self.num_done = 0
        self.progress_bar: tqdm.tqdm = None  # type: ignore[assignment]
        self.max_concurrent_tasks = self.cfg.get("max_concurrent_tasks", 100)

        self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        self.sink: Sink = None  # type: ignore[assignment]
        self.team_manager = ScalableTeamManager(
            self.simulation_id,
            dispatcher_config=self.cfg.get("dispatcher"),
        )
        self.resources: Dict[str, BaseResourceClient] = {}

        random.seed(self.cfg["seed"])
        self.num_trial = self.cfg["num_trial"]
        if self.num_trial > 1:
            self.seeds = [random.randint(0, 2**31 - 1) for _ in range(self.num_trial)]
        else:
            self.seeds = [self.cfg["seed"]]

        self.num_processed = 0
        self.counter_lock = asyncio.Lock()

    async def create_team(
        self,
        cli,
    ):
        """Create team of ray actors from config"""

        # Create sink first - it must exist before other agents
        if "_sink" in self.cfg.agents:
            self.team_manager.create_role(
                role_name="_sink",
                agent_config=self.cfg.agents["_sink"],
                resources=self.resources,
            )

        # Create other roles (they will receive sink reference)
        team: Dict[str, List[ray.actor.ActorHandle]] = {}
        for agent_id, agent_config in self.cfg.agents.items():
            if agent_id == "_sink":
                continue  # Already created
            agents = self.team_manager.create_role(
                role_name=agent_id,
                agent_config=agent_config,
                resources=self.resources,
            )
            team[agent_id] = agents

        # Initialize the team with collected handles
        await self.team_manager.initialize_team(team)
        self.sink = self.team_manager.sink  # type: ignore[assignment]

    async def _progress_task(self):
        async def _update_progress():
            done = await self.sink.get_progress.remote()  # type: ignore[attr-defined]
            if done > self.num_done:
                for _ in range(done - self.num_done):
                    self.semaphore.release()
                async with self.counter_lock:
                    total = self.num_processed
                self.progress_bar.total = total
                self.progress_bar.update(done - self.num_done)
                self.num_done = done

        while self.get_num_inputs() is None or self.num_done < self.get_num_inputs():
            await _update_progress()
            await asyncio.sleep(1)

    async def _producer(
        self, queue: asyncio.Queue, data_items: Generator[Dict[str, Any], None, None]
    ):
        """Producer: adds items to the queue"""
        try:
            count = 0
            for item in data_items:
                for i in range(self.num_trial):
                    await queue.put((i, item))
                    count += 1
        finally:
            logger.info(f"Producer finished: {count} items queued")

    async def _consumer(self, id, queue: asyncio.Queue):
        """Consumer: processes items from the queue"""
        try:
            while True:
                trial_item = await queue.get()
                if trial_item is None:  # Sentinel value to stop
                    break

                try:
                    await self._process_item(trial_item)
                except Exception as e:
                    logger.error(f"Error processing item: {repr(e)}")
                finally:
                    queue.task_done()
                async with self.counter_lock:
                    self.num_processed += 1
        finally:
            logger.debug(f"Consumer_{id} finished")

    async def _process_item(self, trial_item: Tuple[int, Dict[str, Any]]):
        await self.semaphore.acquire()
        logger.debug("Start process_item")
        trial, item = trial_item
        handle = ray.put(item)
        await self.sink.register_object.remote([handle])  # type: ignore[attr-defined]
        orchestrator = instantiate(self.cfg.orchestrator)
        first_agent_role = orchestrator.current_agent()

        try:
            await orchestrator.init(
                self.simulation_id,
                self.team_manager.get_team_config()[first_agent_role],
                self.sink,
                metadata={
                    "trial": trial,
                    "task": item,
                    "seed": self.seeds[trial],
                    "task_ref": handle,
                },
                resources=self.resources,
                logger=logger,
            )
            orchestrator.init_timestamp = time.time()
        except Exception as e:
            traceback.print_exc()
            logger.error(
                f"Error initializing orchestrator for item {orchestrator.id}: {repr(e)}"
            )
            await self.sink.receive_message.remote(orchestrator)  # type: ignore[attr-defined]
            return

        logger.debug(f"done Init {orchestrator.id}")
        logger.debug(f"Enqueue: {orchestrator.id}")

        # Enqueue to the first agent's Dispatcher
        try:
            await self.team_manager.dispatchers[first_agent_role].enqueue.remote(orchestrator)
        except Exception as e:
            logger.error(f"Failed to enqueue to dispatcher for {first_agent_role}: {repr(e)}")
            orchestrator.status["error"] = f"Failed to reach {first_agent_role}: {e}"
            await self.sink.receive_message.remote(orchestrator)  # type: ignore[attr-defined]
            return

        logger.debug(f"Done Enqueue: {orchestrator.id}")

        if self.cfg.get("rate_limit_enqueue", False):
            await asyncio.sleep(20)

    def get_num_inputs(self):
        count = self.data_loader.total_count()
        return (count * self.num_trial) if count else None

    async def run_simulation(self):
        """Run the P2P simulation"""

        setup_logging(logger, self.cfg.get("debug", False))
        logger.info("Config-Driven P2P Agent Simulation")
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(self.cfg, resolve=True)}")
        cli = Cli(**self.cfg.matrix)
        if not ray.is_initialized():
            if os.environ.get("RAY_ADDRESS"):
                # already inside ray
                ray.init()
            else:
                ray.init(
                    address=get_ray_address(cli.cluster.cluster_info()),  # type: ignore[arg-type]
                    log_to_driver=True,
                )

        # Load tasks
        self.data_loader = instantiate(self.cfg.dataset)
        data_items = self.data_loader.load_data()

        if self.cfg.get("resources"):
            for res_id, res_config in self.cfg.resources.items():
                self.resources[res_id] = instantiate(
                    res_config, resource_id=res_id, matrix_cli=cli
                )
        async with AsyncExitStack() as stack:
            self.resources = {
                res_id: await stack.enter_async_context(res)
                for res_id, res in self.resources.items()
            }
            for res in self.resources.values():
                await res.init(self.resources, logger)

            logger.info(f"Resources: {list(self.resources.keys())}")

            # Create team
            await self.create_team(cli)
            await self.sink.set_metrics_output.remote(  # type: ignore[attr-defined]
                self.cfg.get("metrics"), self.cfg.get("output", {})
            )

            progress_future = asyncio.create_task(self._progress_task())

            self.progress_bar = tqdm.tqdm(
                total=self.get_num_inputs(),
                desc=self.simulation_id,
                unit="task",
                disable=self.sim_index > 0,
            )

            logger.info(
                f"Starting P2P simulation {self.simulation_id} (namespace: {self.simulation_id})"
            )
            # Process tasks
            if self.cfg.get("rate_limit_enqueue", False):
                num_consumers = 1
            else:
                num_consumers = min(1000, self.max_concurrent_tasks)
            queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_concurrent_tasks * 2)
            consumers = [
                asyncio.create_task(self._consumer(i, queue))
                for i in range(num_consumers)
            ]
            producer_task = asyncio.create_task(self._producer(queue, data_items))
            await producer_task
            self.progress_bar.total = self.get_num_inputs()
            self.progress_bar.refresh()
            await self.sink.set_num_inputs.remote(self.get_num_inputs())  # type: ignore[attr-defined]
            for _ in range(num_consumers):
                await queue.put(None)
            await asyncio.gather(*consumers, return_exceptions=True)

            # wait for task to finish
            await progress_future

            # Shutdown agents
            await self.team_manager.shutdown()

        overall_metrics = await self.sink.get_overall_metrics.remote()  # type: ignore[attr-defined]
        for metric, value in overall_metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        # Log dead task count if any
        num_dead = await self.sink.get_num_dead.remote()  # type: ignore[attr-defined]
        if num_dead > 0:
            logger.warning(f"Dead/lost tasks: {num_dead}")

        return overall_metrics


@hydra.main(config_path="config", config_name="coral_experiment", version_base=None)
def main(cfg: DictConfig):
    num_tasks = cfg.get("parallelism", 1)
    if num_tasks > 1 and cfg.dataset.get("data_files"):
        setup_logging(logger, cfg.get("debug", False))
        cli = Cli(**cfg.matrix)
        if not ray.is_initialized():
            if os.environ.get("RAY_ADDRESS"):
                # already inside ray
                ray.init()
            else:
                ray.init(
                    address=get_ray_address(cli.cluster.cluster_info()),  # type: ignore[arg-type]
                    log_to_driver=True,
                )

        logger.info(f"Launching {num_tasks} Ray actors for parallel processing")

        # Log cut_off division info
        cut_off = cfg.dataset.get("cut_off", None)
        if cut_off is not None:
            per_job_cut_off = int(cut_off / num_tasks)
            logger.info(
                f"Dividing cut_off {cut_off} by {num_tasks} tasks = {per_job_cut_off} per job"
            )

        # Subsample dataset into chunks
        data_files = sorted(glob.glob(os.path.expanduser(cfg.dataset.data_files)))
        logger.info(
            f"Found {len(data_files)} data files, splitting into {num_tasks} chunks"
        )
        file_chunks = np.array_split(data_files, num_tasks)

        # Launch Ray actors
        P2PAgentFrameworkActor = ray.remote(P2PAgentFramework)
        actors = []

        output_path = Path(cfg.output.path).expanduser()
        parent = output_path.parent
        name = output_path.name

        # Split name into base and extensions
        base_name = name.split(".", 1)[0]  # Get first part before any dot
        extensions = (
            "." + name.split(".", 1)[1] if "." in name else ""
        )  # Get everything after first dot

        for i, paths_split in enumerate(file_chunks):
            paths_split = paths_split.tolist()
            if len(paths_split) == 0:
                continue

            # Create job-specific config
            job_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

            # Update for this job
            OmegaConf.update(job_cfg, "dataset.data_files", paths_split, merge=True)
            split_output = parent / f"{base_name}-split-{i:04d}{extensions}"
            OmegaConf.update(job_cfg, "output.path", split_output, merge=True)
            if cut_off is not None:
                OmegaConf.update(
                    job_cfg, "dataset.cut_off", per_job_cut_off, merge=True
                )

            logger.info(f"Actor {i}: processing {len(paths_split)} files")
            actor = P2PAgentFrameworkActor.remote(i, job_cfg)  # type: ignore[arg-type]
            actors.append(actor)

        # Run all actors in parallel
        futures = [actor.run_simulation.remote() for actor in actors]  # type: ignore

        # Wait for all to complete
        logger.info(f"Waiting for {len(futures)} actors to complete...")
        results = ray.get(futures)

        # Log results
        for i, result in enumerate(results):
            logger.info(f"Actor {i}: {result}")

        logger.info("All Ray actors completed successfully")
    else:
        framework = P2PAgentFramework(0, cfg)
        asyncio.run(framework.run_simulation())


if __name__ == "__main__":
    main()

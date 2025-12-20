# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
import asyncio
import glob
import json
import logging
import os
import random
import time
import traceback
from collections import defaultdict
from contextlib import AbstractAsyncContextManager, AsyncExitStack, nullcontext
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union

import hydra
import numpy as np
import ray
import tqdm
import zstandard as zstd
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from ray.util.metrics import Counter, Gauge, Histogram

from matrix import Cli
from matrix.utils.ray import get_ray_address, ray_get_async

from .agent_utils import get_ray_actor_class, setup_logging

logger = logging.getLogger(__name__)


# ==== Abstract Orchestrator ====
class Orchestrator(abc.ABC):

    def __init__(self):
        self._id = None
        self.resource_state: dict[str, Any] = {}
        self.status: Dict[str, Any] = {}

    async def init(
        self,
        simulation_id: str,
        first_agent: Tuple[Type, DictConfig],
        sink: "Sink",
        metadata: dict[str, Any],
        resources: dict[str, "BaseResourceClient"],
        logger: logging.Logger,
    ) -> None:
        self.simulation_id = simulation_id
        self.trial = metadata["trial"]
        self.seed = metadata["seed"]
        task = metadata["task"]
        self.task_ref: ray.ObjectRef = metadata["task_ref"]

        self.resource_state = {
            res_id: await res.acquire(task, logger) for res_id, res in resources.items()
        }

    @property
    def id(self) -> str:
        return f"{self.simulation_id}_id-{self._id}_trial-{self.trial}"

    def is_success(self) -> bool:
        return self.status.get("success", False)

    @abc.abstractmethod
    def current_agent(self) -> str:
        """Get the current agent's ID."""
        pass

    @abc.abstractmethod
    async def is_done(self) -> bool:
        pass

    @abc.abstractmethod
    async def update(
        self,
        result: Any,
        updater: "AgentActor",
        logger: logging.Logger,
    ) -> "Orchestrator":
        """Update the orchestrator with the agent's result."""
        pass

    @abc.abstractmethod
    async def to_output(self) -> Dict[str, Any]:
        pass

    async def cleanup(
        self, sink: "Sink", resources: dict[str, "BaseResourceClient"], logger
    ):
        for res_id, res in (self.resource_state or {}).items():
            await resources[res_id].release(res, logger)
        self.resource_state = {}
        loop = asyncio.get_event_loop()
        await sink.unregister_object([self.task_ref])  # type: ignore[attr-defined]
        await loop.run_in_executor(None, lambda: ray.internal.free([self.task_ref]))
        self.task_ref = None  # type: ignore[assignment]

    async def get_task(self):
        return await self.task_ref


class BaseResourceClient:
    def __init__(self, resource_id: str):
        self.resource_id = resource_id

    async def init(self, resources: dict[str, "BaseResourceClient"], logger):
        pass

    async def acquire(self, task: Dict[str, Any], logger):
        return None

    async def release(self, resource_info: Any, logger):
        pass

    async def utilize(self, resource_info: Any, logger, **kwargs):
        pass

    async def check_health(self):
        return True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False  # re-raise exception if one happened


# ==== Abstract AgentActor ====
# @ray.remote
class AgentActor(abc.ABC):
    def __init__(
        self,
        id: str,
        agent_id: str,
        config: DictConfig,
        resources: dict[str, BaseResourceClient],
    ):
        # PATCH FIRST - before any HTTP clients are created
        self._patch_getproxies()

        self.id = id
        self.agent_id = agent_id
        self.config = config
        self.resource_name = config.get("resource_name")
        if self.resource_name:
            logger.debug(f"Resources {list(resources.keys())}")
            self.resource_client: Optional[BaseResourceClient] = resources[
                self.resource_name
            ]
        else:
            self.resource_client = None
        self.resources = resources  # used for releasing all resources
        system_prompt = config.get("system_prompt", "")
        self.debug = config.get("debug", False)

        self.queue: asyncio.Queue["Orchestrator"] = asyncio.Queue()
        self.running = True  # should run forever
        self.event_loop_task: None | asyncio.Task = None
        self.pending_tasks: set[asyncio.Task] = set()
        self.system_prompt = system_prompt
        self.logger = logging.getLogger(self.__class__.__name__)
        setup_logging(self.logger, self.debug)

        self._init_metrics()

    @staticmethod
    def _patch_getproxies():
        """Patch urllib to handle concurrent environment modifications in Ray."""
        import os
        import urllib.request

        original_getproxies = urllib.request.getproxies_environment

        def safe_getproxies():
            """Thread-safe version that handles missing keys during iteration."""
            try:
                # Create a snapshot of environment to avoid iteration issues
                env_copy = dict(os.environ)
                proxies = {}
                for name in ["http", "https", "ftp", "no"]:
                    proxy_var = name + "_proxy"
                    if proxy_var in env_copy:
                        proxies[name] = env_copy[proxy_var]
                    elif proxy_var.upper() in env_copy:
                        proxies[name] = env_copy[proxy_var.upper()]
                return proxies
            except (KeyError, RuntimeError):
                # If anything goes wrong, return empty dict
                return {}

        # Apply patch
        urllib.request.getproxies_environment = safe_getproxies
        urllib.request.getproxies = safe_getproxies

    def _init_metrics(self):
        """Initialize Ray metrics for monitoring"""

        default_tags = {
            "id": self.id,
            "agent_id": self.agent_id,
        }
        tag_keys = ("id", "agent_id")

        # Define all metrics in a declarative list
        metrics_config = [
            # (attribute_name, metric_class, name, description, extra_kwargs)
            (
                "messages_processed",
                Counter,
                "agent_messages_processed",
                "Total number of messages processed by this agent",
                {},
            ),
            (
                "queue_size",
                Gauge,
                "agent_queue_size",
                "Current queue size for this agent",
                {},
            ),
            (
                "messages_received",
                Counter,
                "agent_messages_received",
                "Total number of messages received by this agent",
                {},
            ),
            (
                "pending_tasks_count",
                Gauge,
                "agent_pending_tasks_count",
                "Number of tasks currently being processed",
                {},
            ),
            (
                "tasks_started",
                Counter,
                "agent_tasks_started",
                "Total number of tasks started",
                {},
            ),
            (
                "tasks_completed",
                Counter,
                "agent_tasks_completed",
                "Total number of tasks completed",
                {},
            ),
            (
                "task_exceptions",
                Counter,
                "agent_task_exceptions",
                "Total number of task exceptions",
                {},
            ),
            (
                "handle_latency",
                Histogram,
                "agent_handle_latency_seconds",
                "Latency of handling each orchestrator task in seconds",
                {
                    "boundaries": self.config.get(
                        "latency_boundaries", self._get_default_latency_boundaries()
                    )
                },
            ),
        ]

        # Create all metrics from the config
        for (
            attr_name,
            metric_class,
            metric_name,
            description,
            extra_kwargs,
        ) in metrics_config:
            metric = metric_class(
                metric_name, description=description, tag_keys=tag_keys, **extra_kwargs
            )
            metric.set_default_tags(default_tags)
            setattr(self, attr_name, metric)

    def _get_default_latency_boundaries(self):
        """
        Override this method in subclasses to provide agent-specific boundaries.
        Default covers a wide range from sub-second to 30 minutes.
        """
        return [
            0.01,
            0.05,
            0.1,
            0.5,
            1.0,
            5.0,
            10.0,
            30.0,
            60.0,
            120.0,
            300.0,
            600.0,
            1200.0,
            1800.0,
        ]

    async def set_team(self, team: Dict[str, list[ray.actor.ActorHandle]]):
        self.sink = team["_sink"][0]
        self.team = team
        if self.event_loop_task is None:
            self.event_loop_task = asyncio.create_task(self._event_loop())

    def get_resources(self):
        return self.resources

    async def receive_message(self, orchestrator: Orchestrator):
        await self.queue.put(orchestrator)
        self.messages_received.inc()  # type: ignore[attr-defined]
        self.queue_size.set(self.queue.qsize())  # type: ignore[attr-defined]

    async def _event_loop(self):

        async def _handle_task_exception(orchestrator, msg):
            orchestrator._append(
                self.agent_id, {"status_ok": False, "error": msg}, self.sink
            )
            await self.sink.receive_message.remote(orchestrator)

        def _log_exceptions(task):
            try:
                task.result()  # will re-raise the exception if one occurred
            except Exception as e:
                self.task_exceptions.inc()  # type: ignore[attr-defined]
                msg = f"Exception in task for agent {self.agent_id}: {e}"
                self.logger.warning(msg)
                traceback.print_exc()

                # Retrieve the orchestrator from the task
                orchestrator = getattr(task, "_orchestrator")
                asyncio.create_task(_handle_task_exception(orchestrator, msg))
            finally:
                self.tasks_completed.inc()  # type: ignore[attr-defined]
                self.pending_tasks_count.set(len(self.pending_tasks))  # type: ignore[attr-defined]

        while self.running:
            orchestrator = await self.queue.get()
            # Update queue size after getting message
            self.queue_size.set(self.queue.qsize())  # type: ignore[attr-defined]

            task = asyncio.create_task(self._handle(orchestrator))
            # Attach orchestrator to task for error logging
            task._orchestrator = orchestrator  # type: ignore[attr-defined]

            self.pending_tasks.add(task)
            self.tasks_started.inc()  # type: ignore[attr-defined]
            self.pending_tasks_count.set(len(self.pending_tasks))  # type: ignore[attr-defined]

            # Clean up completed tasks
            task.add_done_callback(self.pending_tasks.discard)
            task.add_done_callback(_log_exceptions)

        if self.pending_tasks:
            await asyncio.gather(*self.pending_tasks, return_exceptions=True)

    async def _handle(self, orchestrator: Orchestrator):
        start_time = time.time()

        self.logger.debug(f"Agent {self.agent_id} handling {orchestrator.id}")
        result = await self.preprocess(orchestrator)
        result = await self.process(orchestrator, result)
        result = await self.postprocess(orchestrator, result)
        if self.agent_id != "_sink":
            next_state = await orchestrator.update(result, self, self.logger)
            if await next_state.is_done():
                next_agent = self.sink
            else:
                next_agent_name = next_state.current_agent()
                next_agent = random.choice(self.team[next_agent_name])
            await next_agent.receive_message.remote(next_state)  # type: ignore[attr-defined]
        else:
            await orchestrator.cleanup(self, self.resources, self.logger)  # type: ignore[arg-type]

        # Record latency and increment messages processed counter
        latency = time.time() - start_time
        self.handle_latency.observe(latency)  # type: ignore[attr-defined]
        self.messages_processed.inc()  # type: ignore[attr-defined]

    async def shutdown(self):
        """Gracefully shutdown the agent"""
        self.running = False

    @classmethod
    async def get_task_message(
        self, config: DictConfig, task: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get the initial message for the agent"""
        raise RuntimeError(
            f"{self.__class__.__name__} does not support initial message generation."
        )

    async def check_health(self):
        return True

    @abc.abstractmethod
    async def preprocess(self, orchestrator: Orchestrator) -> Any:
        """Preprocess the orchestrator before sending to LLM or processing"""
        pass

    async def postprocess(self, orchestrator: Orchestrator, response: Any) -> Any:
        """Postprocess the response after LLM or processing"""
        return response

    async def process(self, orchestrator: Orchestrator, response: Any) -> Any:
        """Postprocess the response after LLM or processing"""
        return response


class BaseDatasetLoader(abc.ABC):
    @abc.abstractmethod
    def load_data(self) -> Generator[Dict[str, Any], None, None]:
        """Load data from the dataset"""
        pass

    @abc.abstractmethod
    def total_count(self) -> Optional[int]:
        pass


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


# @ray.remote
class Sink(AgentActor):
    def __init__(
        self,
        id,
        agent_id: str,
        config: DictConfig,
        resources: dict[str, BaseResourceClient],
    ):
        super().__init__(id, agent_id, config, resources)  # type: ignore[arg-type]
        self.num_done = 0
        self.num_inputs: Optional[int] = None
        self.ray_objects: dict[str, ray.ObjectRef] = {}  # hold the ref to avoid gc

    async def set_metrics_output(
        self,
        metrics_cfg: dict[str, Any],
        output_cfg: dict[str, Any],
    ):
        if metrics_cfg:
            self.metrics_accumulator: Optional[BaseMetricsAccumulator] = instantiate(
                metrics_cfg
            )
        else:
            self.metrics_accumulator = None

        self.save_success_only = output_cfg.get("success_only", False)
        self.output_path = os.path.abspath(os.path.expanduser(output_cfg["path"]))
        self.logger.info(f"Output file is {self.output_path}")
        if self.output_path.endswith(".zst"):
            cctx = zstd.ZstdCompressor(level=3)
            self.output_file = cctx.stream_writer(open(self.output_path, "wb"))
        else:
            self.output_file = open(self.output_path, "w", encoding="utf-8")  # type: ignore[assignment]

    async def set_num_inputs(self, num_inputs: int):
        self.num_inputs = num_inputs

    async def preprocess(self, orchestrator: "Orchestrator"):

        def _write_output(output_data, output_path):
            """CPU-intensive work: JSON serialization, encoding, and compression"""
            json_line = json.dumps(output_data, ensure_ascii=False, default=str)

            if output_path.endswith(".zst"):
                return (json_line + "\n").encode("utf-8")
            else:
                return json_line + "\n"

        if not self.save_success_only or orchestrator.is_success():
            # Run CPU-intensive work in thread pool
            loop = asyncio.get_event_loop()
            data_to_write = await loop.run_in_executor(
                None,
                partial(
                    _write_output, await orchestrator.to_output(), self.output_path
                ),
            )
            self.output_file.write(data_to_write)

        self.num_done += 1

        if self.metrics_accumulator:
            self.metrics_accumulator.accumulate(orchestrator)

        if self.num_inputs is not None and self.num_done >= self.num_inputs:
            self.output_file.close()
        return {"orchestrator": orchestrator}

    async def get_progress(self) -> int:
        return self.num_done

    async def get_overall_metrics(self) -> dict[str, Any] | None:
        return (
            self.metrics_accumulator.done()
            if self.metrics_accumulator is not None
            else {}
        )

    async def check_health(self):
        return True

    async def register_object(self, obj: list[ray.ObjectRef]):
        o = obj[0]
        self.ray_objects[o.hex()] = o  # type: ignore[attr-defined]

    async def unregister_object(self, obj: list[ray.ObjectRef]):
        for o in obj:
            self.ray_objects.pop(o.hex(), None)  # type: ignore[attr-defined]


class ScalableTeamManager:
    """Manages teams with multiple actors per role using load balancers when needed"""

    def __init__(self, simulation_id: str):
        self.simulation_id = simulation_id
        self.team: Dict[str, List[ray.actor.ActorHandle]] = {}
        self.teamConfig: Dict[str, Tuple[Type, DictConfig]] = {}

    def create_role(self, role_name: str, agent_config: DictConfig, resources):
        """Create agents for a role. Uses load balancer only if count > 1"""

        # Create agents (your existing AgentActor - no names!)
        agents = []
        count = agent_config.get("num_instances", 1)
        ray_resources: dict[str, Any] = {}
        if "ray_resources" in agent_config:
            ray_resources = OmegaConf.to_container(  # type: ignore[assignment]
                agent_config["ray_resources"], resolve=True
            )

        # Create agent actor - need to get the class first, then create remote
        agent_class = get_ray_actor_class(
            agent_config._target_
        )  # This is already a Ray remote class

        for i in range(count):
            # No agent_id parameter to avoid naming conflicts
            agent = agent_class.options(
                name=f"{role_name}_{i}",  # per-role unique index
                namespace=self.simulation_id,  # simulation id groups them
                **ray_resources,
            ).remote(
                id=f"{self.simulation_id}_{role_name}_{i}",
                agent_id=role_name,
                config=agent_config,
                resources=resources,
            )
            logger.info(f"Created agent: {role_name} id={agent._actor_id.hex()}")
            agents.append(agent)

        self.team[role_name] = agents
        self.teamConfig[role_name] = (
            agent_class.__ray_metadata__.modified_class,
            agent_config,
        )

        return agents

    async def initialize_team(self):
        """Initialize all agents with team references"""

        # Check Ray Actor health
        all_actors = {
            f"{name}_{i}": actor
            for name, actor_list in self.team.items()
            for i, actor in enumerate(actor_list)
        }
        logger.info(f"Checking Ray actor health for {list(all_actors.keys())}")
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    *[handle.check_health.remote() for handle in all_actors.values()]
                ),
                timeout=10 * len(all_actors),
            )
        except Exception as e:
            logger.error(
                f"Failed to start Ray actors, check cluster resource utilization. {repr(e)}"
            )
            raise e
        logger.info("Checking Ray actor health done...")

        # Set team for all individual agents
        for role_name, agents in self.team.items():
            for agent in agents:
                await agent.set_team.remote(self.team)

    def get_team(self):
        """Get team dictionary for orchestrator routing"""
        return self.team

    def get_team_config(self):
        """Get team dictionary for orchestrator routing"""
        return self.teamConfig

    async def shutdown(self):
        for agents in self.team.values():
            for agent in agents:
                await agent.shutdown.remote()


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
        self.team_manager = ScalableTeamManager(self.simulation_id)
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

        for agent_id, agent_config in self.cfg.agents.items():

            self.team_manager.create_role(
                role_name=agent_id,
                agent_config=agent_config,
                resources=self.resources,
            )

        # Initialize the team
        await self.team_manager.initialize_team()
        self.sink = self.team_manager.get_team()["_sink"][0]

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
        first_agent = random.choice(
            self.team_manager.get_team()[orchestrator.current_agent()]
        )
        try:
            await orchestrator.init(
                self.simulation_id,
                self.team_manager.get_team_config()[orchestrator.current_agent()],
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
        except Exception as e:
            traceback.print_exc()
            logger.error(
                f"Error initializing orchestrator for item {orchestrator.id}: {repr(e)}"
            )
            await self.sink.receive_message.remote(orchestrator)  # type: ignore[attr-defined]
            return

        logger.debug(f"done Init {orchestrator.id}")
        logger.debug(f"Enqueue: {orchestrator.id}")
        await first_agent.receive_message.remote(orchestrator)
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

            logger.info(f"Starting P2P simulation {self.simulation_id}")
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

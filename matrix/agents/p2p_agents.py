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
import pickle
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

ENABLE_INSTRUMENTATION = False

# ==== Utility Functions ====
async def send_with_retry(
    orchestrator: "Orchestrator",
    role: str,
    sink: ray.actor.ActorHandle,
    local_cache: Dict[str, List[ray.actor.ActorHandle]],
    log: logging.Logger,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> Dict[str, List[ray.actor.ActorHandle]]:
    """
    Send orchestrator to an agent with local cache and fault-tolerant retry.

    Args:
        orchestrator: The orchestrator state to send
        role: The role name of the target agent
        sink: The sink actor handle for registry lookups
        local_cache: Local team cache dict (will be updated on refresh)
        log: Logger instance for warnings
        timeout: Timeout for actor acquisition
        max_retries: Maximum retry attempts

    Returns:
        Updated local_cache dict

    Raises:
        RuntimeError: If all retries exhausted
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            if role == "_sink":
                agent = sink
            elif attempt == 0 and role in local_cache and local_cache[role]:
                # First attempt: use local cache for speed
                agent = random.choice(local_cache[role])
            else:
                # Fallback: get from sink with force refresh and update local cache
                agent = await sink.get_actor.remote(role, timeout, True)
                local_cache = await sink.get_team_snapshot.remote()

            await agent.receive_message.remote(orchestrator)
            return local_cache  # Success
        except ray.exceptions.RayActorError as e:
            last_exception = e
            log.warning(
                f"Actor {role} is dead (attempt {attempt + 1}/{max_retries}): {repr(e)}"
            )
            # Clear local cache for this role to force refresh
            local_cache.pop(role, None)
            continue
        except TimeoutError as e:
            last_exception = e
            log.warning(
                f"Timeout getting actor {role} (attempt {attempt + 1}/{max_retries}): {repr(e)}"
            )
            continue
        except Exception:
            # For other exceptions, don't retry
            raise

    # All retries exhausted
    raise RuntimeError(
        f"Failed to send to {role} after {max_retries} attempts: {last_exception}"
    )


# ==== Abstract Orchestrator ====
class Orchestrator(abc.ABC):

    def __init__(self):
        self._id = None
        self.resource_state: dict[str, Any] = {}
        self.status: Dict[str, Any] = {}
        self.creation_timestamp = time.time()
        self.init_timestamp = 0.0
        self.finish_timestamp = 0.0
        self.enqueue_timestamp = 0.0
        self.instrumentation: list[tuple[str, str, Any]] = []  # list of key value

    async def to_output(self) -> Dict[str, Any]:
        return {
            "current_agent": self.current_agent(),
            "id": self._id,
            "trial": self.trial,
            "seed": self.seed,
            "task": await self.get_task(),
            "creation_timestamp": self.creation_timestamp,
            "init_timestamp": self.init_timestamp,
            "finish_timestamp": self.finish_timestamp,
            "instrumentation": self.instrumentation,  # temporary
            "history": [
                {"agent": msg.agent, "response": await msg.response.to_dict()}
                for msg in self.history
            ],
            "status": self.status,
        }

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

    def append_instrumentation(self, metric, agent_id, measure):  # temporary
        self.instrumentation.append((metric._name, agent_id, measure))


class DeadOrchestrator(Orchestrator):
    """
    A minimal orchestrator representing a lost/dead task.
    Used to write tombstone records through the normal Sink flow.
    """

    def __init__(self, orchestrator_id: str, error: str = "Actor died while processing this task"):
        super().__init__()
        self._id = orchestrator_id
        self.simulation_id = ""
        self.trial = -1
        self.seed = -1
        self.error = error
        self.task_ref = None  # type: ignore[assignment]
        self.status = {"success": False, "error": error}

    @property
    def id(self) -> str:
        return self._id

    def current_agent(self) -> str:
        return "_sink"

    async def is_done(self) -> bool:
        return True

    async def update(
        self,
        result: Any,
        updater: "AgentActor",
        logger: logging.Logger,
    ) -> "Orchestrator":
        return self

    async def to_output(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "status": "lost",
            "error": self.error,
            "timestamp": self.finish_timestamp or time.time(),
        }

    async def cleanup(
        self, sink: "Sink", resources: dict[str, "BaseResourceClient"], logger
    ):
        # No resources to clean up for dead orchestrator
        pass

    async def get_task(self):
        return None


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
    THROUGHPUT_WINDOWS = 60  # latency in seconds to measure throughput

    def __init__(
        self,
        id: str,
        agent_id: str,
        config: DictConfig,
        resources: dict[str, BaseResourceClient],
        sink: Optional[ray.actor.ActorHandle] = None,
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
        self.pending_tasks: set[asyncio.Task] = set()
        # Track last message time for idle detection
        self.last_message_time: float = time.time()
        self.system_prompt = system_prompt
        self.logger = logging.getLogger(self.__class__.__name__)
        setup_logging(self.logger, self.debug)

        # Store sink reference and start event loop
        # For Sink actor itself, sink will be None (set later via _set_self_as_sink)
        self.sink = sink
        # Local team cache for fast actor lookup, updated from sink on failure
        self._local_team_cache: Dict[str, List[ray.actor.ActorHandle]] = {}

        if self.sink is not None:
            # Start event loop immediately since we have sink reference
            self.event_loop_task: Optional[asyncio.Task] = (
                asyncio.get_event_loop().create_task(self._event_loop())
            )
        else:
            self.event_loop_task = None

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
                Gauge,
                "agent_handle_latency_seconds",
                "Latency of handling each orchestrator task in seconds",
                {},
            ),
            (
                "dequeue_latency",
                Gauge,
                "dequeue_latency_seconds",
                "Time staying in queue in seconds",
                {},
            ),
            (
                "throughput",
                Gauge,
                "throughput",
                "num of messages processed in the last window",
                {},
            ),
            # temporary instrumentation
            (
                "ser_size_kb",
                Gauge,
                "ser_size_kb",
                "num of kb for the serialized message object",
                {},
            ),
        ]
        self._init_metrics(metrics_config)
        self.throughput_helper = {
            "cur_messages_processed": 0,
            "last_timestamp": time.time(),
            "last_messages_processed": 0,
        }

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

    def _init_metrics(self, metrics_config):
        """Initialize Ray metrics for monitoring"""

        default_tags = {
            "id": self.id,
            "agent_id": self.agent_id,
        }
        tag_keys = ("id", "agent_id")

        # Define all metrics in a declarative list

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

    def get_resources(self):
        return self.resources

    async def receive_message(self, orchestrator: Orchestrator):
        now = time.time()
        orchestrator.enqueue_timestamp = now
        self.last_message_time = now  # Update for idle detection
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
            latency = time.time() - orchestrator.enqueue_timestamp
            self.dequeue_latency.set(latency)
            if ENABLE_INSTRUMENTATION:
                orchestrator.append_instrumentation(
                    self.dequeue_latency, self.agent_id, latency
                )

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
        start_time = time.perf_counter()

        self.logger.debug(f"Agent {self.agent_id} handling {orchestrator.id}")
        result = await self.preprocess(orchestrator)
        result = await self.process(orchestrator, result)
        result = await self.postprocess(orchestrator, result)
        if self.agent_id != "_sink":
            next_state = await orchestrator.update(result, self, self.logger)
            if await next_state.is_done():
                next_agent_name = "_sink"
            else:
                next_agent_name = next_state.current_agent()

            # temporary
            if ENABLE_INSTRUMENTATION:
                blob = pickle.dumps(next_state)
                size_kb = len(blob) / 1024
                self.ser_size_kb.set(size_kb)
                next_state.append_instrumentation(
                    self.ser_size_kb, self.agent_id, (time.time(), size_kb)
                )
                latency = time.perf_counter() - start_time
                next_state.append_instrumentation(
                    self.handle_latency, self.agent_id, latency
                )

            # Send to next agent with fault-tolerant retry
            self._local_team_cache = await send_with_retry(
                next_state,
                next_agent_name,
                self.sink,
                self._local_team_cache,
                self.logger,
            )
            # Update last message time after successful send
            self.last_message_time = time.time()
        else:
            await orchestrator.cleanup(self, self.resources, self.logger)  # type: ignore[arg-type]

        # Record latency and increment messages processed counter
        latency = time.perf_counter() - start_time
        self.handle_latency.set(latency)  # type: ignore[attr-defined]
        self.messages_processed.inc()  # type: ignore[attr-defined]
        self.throughput_helper["cur_messages_processed"] += 1
        now = time.time()
        if now - self.throughput_helper["last_timestamp"] > self.THROUGHPUT_WINDOWS:
            processed = self.throughput_helper["cur_messages_processed"]
            throughput = (
                processed - self.throughput_helper["last_messages_processed"]
            ) / (now - self.throughput_helper["last_timestamp"])
            self.throughput_helper["last_timestamp"] = now
            self.throughput_helper["last_messages_processed"] = processed
            self.throughput.set(throughput)

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

    async def get_active_count(self) -> Tuple[int, float]:
        """Return the count of active tasks and last message timestamp."""
        return (self.queue.qsize() + len(self.pending_tasks), self.last_message_time)

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
    # Constants for team registry
    REFRESH_INTERVAL = 30.0  # seconds between periodic refreshes
    GET_ACTOR_TIMEOUT = 60.0  # default timeout for get_actor calls
    GET_ACTOR_RETRY_INTERVAL = 1.0  # seconds between retries
    IDLE_TIMEOUT = 60.0  # seconds of idle time before checking for dead tasks

    def __init__(
        self,
        id,
        agent_id: str,
        config: DictConfig,
        resources: dict[str, BaseResourceClient],
    ):
        # Sink is its own sink - pass self to parent (self is valid before super())
        super().__init__(id, agent_id, config, resources, sink=self)  # type: ignore[arg-type]

        self.num_done = 0
        self.num_inputs: Optional[int] = None
        self.ray_objects: dict[str, ray.ObjectRef] = {}  # hold the ref to avoid gc

        # Team registry: Sink is the source of truth for actor handles
        self._team: Dict[str, List[ray.actor.ActorHandle]] = {}
        self._team_config: Dict[str, Tuple[int, str]] = {}  # role -> (count, namespace)
        self._team_lock = asyncio.Lock()
        self._refresh_task: Optional[asyncio.Task] = None

        # Track all in-flight orchestrator IDs (added when sent to first agent, removed on completion)
        self.inflight_ids: set[str] = set()

        # Idle detection for dead task recovery
        self.last_message_time: float = time.time()
        self._idle_check_task: Optional[asyncio.Task] = None
        self.num_dead: int = 0  # Counter for dead/lost orchestrators

        additional_metrics_config = [
            (
                "task_init_latency",
                Gauge,
                "task_init_latency_seconds",
                "task init latency",
                {},
            ),
            (
                "e2e_latency",
                Gauge,
                "e2e_latency_seconds",
                "end to end task latency",
                {},
            ),
            (
                "sink_write_latency",
                Gauge,
                "sink_write_latency_seconds",
                "latency to accmulate overall metrics",
                {},
            ),
        ]
        self._init_metrics(additional_metrics_config)

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
        # Start idle check task now that we know the total inputs
        if self._idle_check_task is None:
            self._idle_check_task = asyncio.create_task(self._idle_check_loop())

    async def preprocess(self, orchestrator: "Orchestrator"):
        # Update last message time for idle detection
        self.last_message_time = time.time()

        def _write_output(output_data, output_path):
            """CPU-intensive work: JSON serialization, encoding, and compression"""
            json_line = json.dumps(output_data, ensure_ascii=False, default=str)

            if output_path.endswith(".zst"):
                return (json_line + "\n").encode("utf-8")
            else:
                return json_line + "\n"

        now = time.time()
        orchestrator.finish_timestamp = now
        latency = now - orchestrator.creation_timestamp
        self.e2e_latency.set(latency)

        # Always write tombstones (DeadOrchestrator), otherwise respect save_success_only
        is_tombstone = isinstance(orchestrator, DeadOrchestrator)
        if not self.save_success_only or orchestrator.is_success():
            # Run CPU-intensive work in thread pool
            start_time = time.perf_counter()
            loop = asyncio.get_event_loop()
            data_to_write = await loop.run_in_executor(
                None,
                partial(
                    _write_output, await orchestrator.to_output(), self.output_path
                ),
            )
            self.output_file.write(data_to_write)
            self.sink_write_latency.set(time.perf_counter() - start_time)
        # Remove from inflight after writing to disk
        self.inflight_ids.discard(orchestrator.id)
        self.num_done += 1
        if is_tombstone:
            self.num_dead += 1

        # Skip metrics accumulation for tombstones
        if self.metrics_accumulator and not is_tombstone:
            self.metrics_accumulator.accumulate(orchestrator)

        latency = orchestrator.init_timestamp - orchestrator.creation_timestamp
        self.task_init_latency.set(latency)

        if self.num_inputs is not None and self.num_done >= self.num_inputs:
            self.output_file.close()
        return {"orchestrator": orchestrator}

    async def get_progress(self) -> int:
        return self.num_done

    async def register_inflight(self, orchestrator_id: str):
        """Register an orchestrator as in-flight (before sending to first agent)."""
        self.inflight_ids.add(orchestrator_id)

    async def get_overall_metrics(self) -> dict[str, Any] | None:
        return (
            self.metrics_accumulator.done()
            if self.metrics_accumulator is not None
            else {}
        )

    async def get_num_dead(self) -> int:
        """Return the count of dead/lost orchestrators."""
        return self.num_dead

    async def check_health(self):
        return True

    async def shutdown(self):
        """Gracefully shutdown the Sink agent and cancel background tasks."""
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            self._refresh_task = None
        if self._idle_check_task is not None:
            self._idle_check_task.cancel()
            try:
                await self._idle_check_task
            except asyncio.CancelledError:
                pass
            self._idle_check_task = None
        await super().shutdown()

    async def shutdown_all(self):
        """Shutdown all actors in the team registry, then shutdown self."""
        async with self._team_lock:
            for role, actors in self._team.items():
                for actor in actors:
                    try:
                        await actor.shutdown.remote()
                    except Exception as e:
                        self.logger.warning(f"Error shutting down {role}: {repr(e)}")
        # Finally shutdown self
        await self.shutdown()

    async def register_object(self, obj: list[ray.ObjectRef]):
        o = obj[0]
        self.ray_objects[o.hex()] = o  # type: ignore[attr-defined]

    async def unregister_object(self, obj: list[ray.ObjectRef]):
        for o in obj:
            self.ray_objects.pop(o.hex(), None)  # type: ignore[attr-defined]

    # ==== Team Registry Methods ====
    async def set_team_registry(
        self,
        team_config: Dict[str, Tuple[int, str]],
        initial_team: Optional[Dict[str, List[ray.actor.ActorHandle]]] = None,
    ):
        """
        Initialize the team registry in Sink.

        Args:
            team_config: Dict mapping role -> (count, namespace) for actor lookup
            initial_team: Optional pre-verified actor handles to avoid initial lookup race
        """
        async with self._team_lock:
            self._team_config = team_config
            self._team = initial_team if initial_team is not None else {}

        # Only do initial refresh if we didn't get pre-verified handles
        if initial_team is None:
            await self._refresh_team_internal()

        # Start periodic refresh task
        if self._refresh_task is None:
            self._refresh_task = asyncio.create_task(self._periodic_refresh())
        self.logger.info(
            f"Team registry initialized with roles: {list(team_config.keys())}"
        )

    async def _periodic_refresh(self):
        """Background task to periodically refresh actor handles."""
        while self.running:
            try:
                await asyncio.sleep(self.REFRESH_INTERVAL)
                await self._refresh_team_internal()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(f"Error in periodic team refresh: {repr(e)}")

    async def _refresh_team_internal(self):
        """Internal method to refresh all actor handles from Ray."""
        async with self._team_lock:
            for role, (count, namespace) in self._team_config.items():
                new_handles = []
                for i in range(count):
                    actor_name = f"{role}_{i}"
                    try:
                        handle = ray.get_actor(actor_name, namespace=namespace)
                        new_handles.append(handle)
                    except ValueError:
                        # Actor not found (not yet restarted)
                        self.logger.warning(
                            f"Actor {actor_name} not found in namespace {namespace}"
                        )
                if new_handles:
                    self._team[role] = new_handles
            self.logger.debug(f"Team refreshed: {list(self._team.keys())}")

    async def force_refresh(self):
        """Force an immediate refresh of all actor handles."""
        await self._refresh_team_internal()

    async def _write_tombstone(self, orchestrator_id: str):
        """
        Write a tombstone record for a lost orchestrator via normal Sink flow.

        Args:
            orchestrator_id: The ID of the lost orchestrator
        """
        self.logger.debug(f"Creating tombstone for lost orchestrator: {orchestrator_id}")

        # Create a DeadOrchestrator and process it through normal flow
        dead_orch = DeadOrchestrator(orchestrator_id)
        await self.receive_message(dead_orch)

    async def _idle_check_loop(self):
        """Background task to detect idle state and write tombstones for lost tasks."""
        while self.running:
            try:
                await asyncio.sleep(10.0)  # Check every 10 seconds

                # Skip if we're done
                if self.num_inputs is not None and self.num_done >= self.num_inputs:
                    break

                # Check if we've been idle long enough
                idle_time = time.time() - self.last_message_time
                if idle_time < self.IDLE_TIMEOUT:
                    continue

                # Check if all actors have empty queues and no pending tasks
                all_idle = await self._check_all_actors_idle()
                if not all_idle:
                    continue

                # All actors are idle and we have in-flight tasks - they must be lost
                if self.inflight_ids:
                    self.logger.warning(
                        f"System idle for {idle_time:.1f}s with {len(self.inflight_ids)} "
                        f"in-flight tasks. Writing tombstones."
                    )
                    for orch_id in list(self.inflight_ids):
                        await self._write_tombstone(orch_id)

                    # Wait for tombstones to be processed through the queue
                    while self.queue.qsize() > 0 or len(self.pending_tasks) > 0:
                        await asyncio.sleep(0.1)

                    # Check if we're still short of expected tasks
                    if self.num_inputs is not None and self.num_done < self.num_inputs:
                        self.logger.error(
                            f"Completed {self.num_done} tasks but expected {self.num_inputs}. "
                            f"Some tasks may have been lost before registration."
                        )
                    break

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(f"Error in idle check loop: {repr(e)}")

    async def _check_all_actors_idle(self) -> bool:
        """Check if all actors have zero queue + pending tasks and idle timestamps."""
        now = time.time()
        async with self._team_lock:
            all_actors = [
                actor
                for actors in self._team.values()
                for actor in actors
            ]

        for actor in all_actors:
            try:
                count, last_msg_time = await asyncio.wait_for(
                    actor.get_active_count.remote(),
                    timeout=5.0,
                )
                if count > 0:
                    return False
                # Also check if actor's last message time is within IDLE_TIMEOUT
                if now - last_msg_time < self.IDLE_TIMEOUT:
                    return False
            except Exception as e:
                # If we can't reach an actor, it might be dead - don't consider system idle
                self.logger.debug(f"Failed to get active count from actor: {repr(e)}")
                return False

        return True

    async def get_actor(
        self, role: str, timeout: Optional[float] = None, force_refresh: bool = False
    ) -> ray.actor.ActorHandle:
        """
        Get a random actor handle for the given role.
        Blocks until an actor is available or timeout expires.

        Args:
            role: The role name to get an actor for
            timeout: Maximum time to wait in seconds (default: GET_ACTOR_TIMEOUT)
            force_refresh: If True, refresh handles before returning

        Returns:
            A random actor handle for the role

        Raises:
            TimeoutError: If no actor is available within timeout
            KeyError: If role is not registered
        """
        if timeout is None:
            timeout = self.GET_ACTOR_TIMEOUT

        if force_refresh:
            await self._refresh_team_internal()

        start_time = time.time()
        while True:
            async with self._team_lock:
                if role in self._team and self._team[role]:
                    return random.choice(self._team[role])

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Timeout waiting for actor with role '{role}' after {timeout}s"
                )

            # Wait and retry
            self.logger.debug(
                f"No actor available for role '{role}', retrying in {self.GET_ACTOR_RETRY_INTERVAL}s"
            )
            await asyncio.sleep(self.GET_ACTOR_RETRY_INTERVAL)
            await self._refresh_team_internal()

    async def get_team_snapshot(self) -> Dict[str, List[ray.actor.ActorHandle]]:
        """Get a snapshot of the current team map."""
        async with self._team_lock:
            return dict(self._team)


class ScalableTeamManager:
    """Manages teams with multiple actors per role using load balancers when needed"""

    def __init__(self, simulation_id: str):
        self.simulation_id = simulation_id
        self.teamConfig: Dict[str, Tuple[Type, DictConfig]] = {}
        # Team registry config: role -> (count, namespace) for actor lookup
        self.team_registry_config: Dict[str, Tuple[int, str]] = {}
        # Sink actor handle - must be created first
        self.sink: Optional[ray.actor.ActorHandle] = None

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

        agents = []
        for i in range(count):
            kwargs = {
                "id": f"{self.simulation_id}_{role_name}_{i}",
                "agent_id": role_name,
                "config": agent_config,
                "resources": resources,
            }
            if not is_sink:
                kwargs["sink"] = self.sink

            agent = agent_class.options(
                name=f"{role_name}_{i}",
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

        self.teamConfig[role_name] = (
            agent_class.__ray_metadata__.modified_class,
            agent_config,
        )
        # Only add non-sink roles to registry (sink won't restart)
        if not is_sink:
            self.team_registry_config[role_name] = (count, self.simulation_id)

        return agents

    async def initialize_team(self, team: Dict[str, List[ray.actor.ActorHandle]]):
        """Initialize all agents with team references.

        Args:
            team: Dict mapping role -> list of actor handles (from create_role return values)
        """
        # Use handles from create_role directly (avoid race with ray.get_actor)
        all_actors = [self.sink]
        for role_handles in team.values():
            all_actors.extend(role_handles)

        logger.info(f"Checking Ray actor health for {len(all_actors)} actors")
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    *[handle.check_health.remote() for handle in all_actors]
                ),
                timeout=10 * len(all_actors),
            )
        except Exception as e:
            logger.error(
                f"Failed to start Ray actors, check cluster resource utilization. {repr(e)}"
            )
            raise e
        logger.info("Checking Ray actor health done...")

        # Initialize Sink's team registry with verified handles (avoid re-lookup race)
        await self.sink.set_team_registry.remote(self.team_registry_config, team)

    def get_team_config(self):
        """Get team config dictionary for orchestrator routing"""
        return self.teamConfig

    async def shutdown(self):
        """Shutdown all actors via sink's registry"""
        if self.sink:
            await self.sink.shutdown_all.remote()


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

        # Local team cache for latency-sensitive _process_item
        # Updated from sink on actor failure
        self._local_team_cache: Dict[str, List[ray.actor.ActorHandle]] = {}

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
        self.sink = self.team_manager.sink

        # Initialize local team cache from sink
        self._local_team_cache = await self.sink.get_team_snapshot.remote()

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

        # Register as in-flight before sending to first agent
        await self.sink.register_inflight.remote(orchestrator.id)

        # Send to first agent with local cache for latency, fallback to sink on error
        try:
            self._local_team_cache = await send_with_retry(
                orchestrator,
                first_agent_role,
                self.sink,
                self._local_team_cache,
                logger,
            )
        except RuntimeError as e:
            # All retries exhausted - send to sink as error
            logger.error(str(e))
            orchestrator.status["error"] = f"Failed to reach {first_agent_role}: {e}"
            await self.sink.receive_message.remote(orchestrator)
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

            logger.info(f"Starting P2P simulation {self.simulation_id} (namespace: {self.simulation_id})")
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
    global ENABLE_INSTRUMENTATION
    num_tasks = cfg.get("parallelism", 1)
    ENABLE_INSTRUMENTATION = cfg.get("instrumentation", False)

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

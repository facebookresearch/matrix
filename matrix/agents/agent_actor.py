# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
import asyncio
import logging
import pickle
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import ray
from omegaconf import DictConfig
from ray.util.metrics import Counter, Gauge

from .agent_utils import remote_call_with_retry, setup_logging
from .orchestrator import BaseResourceClient, DeadOrchestrator, Orchestrator

logger = logging.getLogger(__name__)


# ==== Abstract AgentActor ====
# @ray.remote
class AgentActor(abc.ABC):
    ENABLE_INSTRUMENTATION = False

    def __init__(
        self,
        id: str,
        agent_id: str,
        config: DictConfig,
        resources: dict[str, BaseResourceClient],
        sink: ray.actor.ActorHandle,
        dispatcher_name: str = None,
        namespace: str = None,
        ray_name: str = None,
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
        # Dispatcher name + namespace for name-based resolution (None for Sink)
        self.dispatcher_name = dispatcher_name
        self.namespace = namespace
        self.ray_name = ray_name  # This agent's Ray actor name (for dispatcher identification)
        self.dispatcher = None  # resolved lazily in _event_loop

        self.event_loop_task: Optional[asyncio.Task] = (
            asyncio.get_event_loop().create_task(self._event_loop())
        )

        metrics_config: list[tuple[str, type, str, str, dict[str, Any]]] = [
            # (attribute_name, metric_class, name, description, extra_kwargs)
            (
                "messages_processed",
                Counter,
                "agent_messages_processed",
                "Total number of messages processed by this agent",
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
            # temporary instrumentation
            (
                "ser_size_kb",
                Gauge,
                "ser_size_kb",
                "num of kb for the serialized message object",
                {},
            ),
            (
                "queue_size",
                Gauge,
                "agent_queue_size",
                "Current queue size for this agent",
                {},
            ),
        ]
        self._init_metrics(metrics_config)

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
            if self.dispatcher is not None:
                try:
                    await remote_call_with_retry(
                        self.dispatcher.submit_error,
                        orchestrator,
                        orchestrator.id,
                        logger=self.logger,
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to submit error to dispatcher for {orchestrator.id}: {repr(e)}"
                    )
                    # Fall back to sending directly to sink
                    try:
                        await remote_call_with_retry(
                            self.sink.receive_message, orchestrator, logger=self.logger
                        )
                    except Exception:
                        self.logger.error(
                            f"Failed to send error orch {orchestrator.id} to sink, dropping"
                        )
            else:
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

        # Resolve dispatcher for submit routing (if this agent has one)
        if self.dispatcher_name is not None:
            try:
                self.dispatcher = ray.get_actor(self.dispatcher_name, namespace=self.namespace)
            except Exception as e:
                self.logger.error(
                    f"Agent {self.id} failed to find dispatcher {self.dispatcher_name}: {repr(e)}"
                )
                return
            try:
                self_handle = ray.get_runtime_context().current_actor
                await self.dispatcher.agent_started.remote(self.ray_name, self_handle)
            except Exception as e:
                self.logger.error(
                    f"Agent {self.id} failed to call agent_started on dispatcher: {repr(e)}"
                )
                return

        # All agents: read from local queue (dispatcher pushes here, or direct for Sink)
        while self.running:
            orchestrator = await self.queue.get()
            if orchestrator is None:  # Shutdown sentinel
                break
            latency = time.time() - orchestrator.enqueue_timestamp
            self.dequeue_latency.set(latency)  # type: ignore[attr-defined]
            if self.ENABLE_INSTRUMENTATION:
                orchestrator.append_instrumentation(
                    self.dequeue_latency, self.agent_id, latency  # type: ignore[attr-defined]
                )

            self.queue_size.set(self.queue.qsize())  # type: ignore[attr-defined]

            task = asyncio.create_task(self._handle(orchestrator))
            task._orchestrator = orchestrator  # type: ignore[attr-defined]

            self.pending_tasks.add(task)
            self.tasks_started.inc()  # type: ignore[attr-defined]
            self.pending_tasks_count.set(len(self.pending_tasks))  # type: ignore[attr-defined]

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

            # temporary
            if self.ENABLE_INSTRUMENTATION:
                blob = pickle.dumps(next_state)
                size_kb = len(blob) / 1024
                self.ser_size_kb.set(size_kb)  # type: ignore[attr-defined]
                next_state.append_instrumentation(
                    self.ser_size_kb, self.agent_id, (time.time(), size_kb)  # type: ignore[attr-defined]
                )
                latency = time.perf_counter() - start_time
                next_state.append_instrumentation(
                    self.handle_latency, self.agent_id, latency  # type: ignore[attr-defined]
                )

            if self.dispatcher is not None:
                # Submit to dispatcher for deterministic routing
                try:
                    await remote_call_with_retry(
                        self.dispatcher.submit,
                        next_state,
                        orchestrator.id,
                        logger=self.logger,
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to submit orch {orchestrator.id} to dispatcher: {repr(e)}"
                    )
                    dead = DeadOrchestrator(
                        orchestrator.id,
                        error=f"Dispatcher unreachable: {e}",
                    )
                    try:
                        await remote_call_with_retry(
                            self.sink.receive_message, dead, logger=self.logger
                        )
                    except Exception:
                        self.logger.error(
                            f"Failed to tombstone orch {orchestrator.id} to sink, dropping"
                        )
            # Update last message time after successful send
            self.last_message_time = time.time()
        else:
            await orchestrator.cleanup(self, self.resources, self.logger)  # type: ignore[arg-type]

        # Record latency and increment messages processed counter
        latency = time.perf_counter() - start_time
        self.handle_latency.set(latency)  # type: ignore[attr-defined]
        self.messages_processed.inc()  # type: ignore[attr-defined]

    async def shutdown(self):
        """Gracefully shutdown the agent"""
        self.running = False
        if self.event_loop_task is not None:
            await self.queue.put(None)  # type: ignore[arg-type]
            await self.event_loop_task

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


class ContainerExecutionAgent(AgentActor):

    @abc.abstractmethod
    async def get_commands(
        self, orchestrator: Orchestrator
    ) -> List[Union[str, List[str]]]:
        pass

    async def preprocess(self, orchestrator: Orchestrator) -> Dict[str, Any]:  # type: ignore[override]
        return {"cmd": await self.get_commands(orchestrator)}

    async def process(self, orchestrator: Orchestrator, response: Any) -> Any:
        cmds = response["cmd"]
        results = []
        assert (
            self.resource_client is not None and orchestrator.resource_state is not None
        )
        for cmd in cmds:
            result = await self.resource_client.utilize(
                orchestrator.resource_state[self.resource_name],
                self.logger,
                cmd=cmd,
            )
            if "returncode" not in result:
                raise Exception(result)
            results.append(result)
        return {"results": results}


class LLMAgentActor(AgentActor):

    async def process(self, orchestrator: Orchestrator, response: Any) -> Any:
        assert self.resource_client is not None
        response = await self.resource_client.utilize(
            {},
            self.logger,
            messages=response,
            task_id=orchestrator.id,
            seed=orchestrator.seed,
        )
        return response

    async def postprocess(self, orchestrator: Orchestrator, response: Any) -> Any:  # type: ignore[override]
        if "text" in response and isinstance(response["text"], list):
            response["text"] = response["text"][0]
        if "tool_calls" in response:
            tool_calls = response["tool_calls"][0]  # because n == 1
            # because openai want a different input tool_call than the one it returns!
            for call in tool_calls:
                call["type"] = "function"
                call["function"] = {
                    "name": call["name"],
                    "arguments": call["arguments"],
                }
                call.pop("name")
                call.pop("arguments")
            response["tool_calls"] = tool_calls
        response["status_ok"] = "error" not in response
        return response

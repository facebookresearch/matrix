# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import ray
from omegaconf import DictConfig

from .agent_utils import HistPair, RayDict

if TYPE_CHECKING:
    from .agent_actor import AgentActor
    from .sink import Sink

logger = logging.getLogger(__name__)


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
        # List of {"agent": str, "response": {"text": str, "tool_calls": [], "tool_call_id": id, "usage": {}, "extracted_answer": str, "status_ok": bool, "agreement": bool}}
        self.history: list[HistPair] = []

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

        # Add initial message to history
        cls, agent_config = first_agent
        initial_message = await cls.get_task_message(agent_config, metadata["task"])  # type: ignore[attr-defined]
        if logger is not None:
            logger.debug(f"Get initial messageMetadataStart {self.id}")
        if initial_message is not None:
            await self._append(
                initial_message["agent"], initial_message["response"], sink
            )

    @property
    def id(self) -> str:
        return f"{self.simulation_id}_id-{self._id}_trial-{self.trial}"

    def is_success(self) -> bool:
        return self.status.get("success", False)

    def is_error(self) -> bool:
        """Return True if the last response had an error (status_ok=False)."""
        if not self.history:
            return True
        return not self.history[-1].response.get("status_ok", True)

    @abc.abstractmethod
    def current_agent(self) -> str:
        """Get the current agent's ID."""
        pass

    @abc.abstractmethod
    async def is_done(self) -> bool:
        pass

    async def update(
        self,
        result: Any,
        updater: "AgentActor",
        logger: logging.Logger,
    ) -> "Orchestrator":
        """Update the orchestrator with the agent's result and determine the next agent."""
        if not isinstance(result, list):
            result = [result]
        for res in result:
            await self._append(self.current_agent(), res, updater.sink)  # type: ignore[arg-type]
        return self

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
        # Cleanup history
        await asyncio.gather(
            *[hist.response.cleanup_ray(sink) for hist in self.history]
        )

    async def get_task(self):
        return await self.task_ref

    def append_instrumentation(self, metric, agent_id, measure):  # temporary
        self.instrumentation.append((metric._name, agent_id, measure))

    async def _append(self, agent: str, msg: dict[str, Any], sink: "Sink"):
        if "timestamp" not in msg:
            msg["timestamp"] = time.time()
        self.history.append(
            HistPair(agent=agent, response=await RayDict.from_dict(msg, sink))
        )


class DeadOrchestrator(Orchestrator):
    """
    A minimal orchestrator representing a lost/dead task.
    Used to write tombstone records through the normal Sink flow.
    """

    def __init__(
        self, orchestrator_id: str, error: str = "Actor died while processing this task"
    ):
        super().__init__()
        self._id = orchestrator_id
        self.simulation_id = ""
        self.trial = -1
        self.seed = -1
        self.error = error
        self.task_ref = None  # type: ignore[assignment]
        self.status = {"error": error}

    @property
    def id(self) -> str:
        return self._id  # type: ignore[return-value]

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


class SequentialOrchestrator(Orchestrator):

    def __init__(
        self,
        interaction_order: List[str],
    ):
        super().__init__()
        self.interaction_order = interaction_order
        self._current_agent_index = 0

    def current_agent(self) -> str:
        if not self.interaction_order:
            raise ValueError("No interaction order defined")
        return self.interaction_order[self._current_agent_index]

    async def update(
        self,
        result: Any,
        updater: "AgentActor",
        logger: logging.Logger,
    ) -> "Orchestrator":
        """Update the orchestrator with the agent's result and determine the next agent."""
        await super().update(result, updater, logger)
        self._current_agent_index = (self._current_agent_index + 1) % len(
            self.interaction_order
        )
        return self

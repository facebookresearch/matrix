# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import os
import time
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, Optional

import ray
import zstandard as zstd
from hydra.utils import instantiate
from omegaconf import DictConfig
from ray.util.metrics import Gauge

from .agent_actor import AgentActor
from .orchestrator import (
    BaseResourceClient,
    DeadOrchestrator,
    Orchestrator,
)

if TYPE_CHECKING:
    from .p2p_agents import BaseMetricsAccumulator

logger = __import__("logging").getLogger(__name__)


# @ray.remote
class Sink(AgentActor):

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
        self.num_dead: int = 0  # Counter for dead/lost orchestrators

        additional_metrics_config: list[tuple[str, type, str, str, dict[str, Any]]] = [
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

    async def preprocess(self, orchestrator: "Orchestrator"):
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
            self.sink_write_latency.set(time.perf_counter() - start_time)  # type: ignore[attr-defined]

        # Increment num_done for ALL arrivals (normal + dead)
        self.num_done += 1

        if is_tombstone:
            self.num_dead += 1
        else:
            self.e2e_latency.set(latency)  # type: ignore[attr-defined]
            latency = orchestrator.init_timestamp - orchestrator.creation_timestamp
            self.task_init_latency.set(latency)  # type: ignore[attr-defined]

            if self.metrics_accumulator:
                self.metrics_accumulator.accumulate(orchestrator)

        # Close output file when all tasks are done and no pending writes
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

    async def get_num_dead(self) -> int:
        """Return the count of dead/lost orchestrators."""
        return self.num_dead

    async def check_health(self):
        return True

    async def shutdown(self):
        """Gracefully shutdown the Sink agent."""
        await super().shutdown()

    async def register_object(self, obj: list[ray.ObjectRef]):
        o = obj[0]
        self.ray_objects[o.hex()] = o  # type: ignore[attr-defined]

    async def unregister_object(self, obj: list[ray.ObjectRef]):
        for o in obj:
            self.ray_objects.pop(o.hex(), None)  # type: ignore[attr-defined]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import os
import random
import time
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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
    # Constants for team registry
    REFRESH_INTERVAL = 30.0  # seconds between periodic refreshes
    GET_ACTOR_TIMEOUT = 60.0  # default timeout for get_actor calls
    GET_ACTOR_RETRY_INTERVAL = 1.0  # seconds between retries

    # dead detection
    IDLE_TIMEOUT = 60.0  # seconds of idle time before checking for dead tasks
    MAX_NEW_ZOMBIE = 10  # at most mark this number at once
    LATE_ARRIVAL_INCR = 5  # make it harder to mark zombie for each late arrivals

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
        self.pending_writes: int = (
            0  # Track in-progress writes to avoid closing file prematurely
        )

        # Team registry: Sink is the source of truth for actor handles
        self._team: Dict[str, List[ray.actor.ActorHandle]] = {}
        self._team_config: Dict[str, Tuple[int, str]] = {}  # role -> (count, namespace)
        self._team_lock = asyncio.Lock()
        self._refresh_task: Optional[asyncio.Task] = None

        # Optimistic timeout tracking for dead orchestrator detection
        self.dead_orchestrator_tracking = config.dead_orchestrator_tracking
        self.dead_order_window = config.max_concurrent_tasks
        # Registration order tracking: when task N completes, tasks before N-max_concurrent_tasks are zombies
        self.registration_counter: int = 0
        self.inflight_orchestrators: dict[str, int] = {}  # id -> registration_order
        self.inflight_order: list[tuple[int, str]] = (
            []
        )  # sorted (order, id) for efficient oldest iteration, lazy update, have stale items
        self.zombie_orchestrators: set[str] = set()  # set of zombie orchestrator ids

        # Idle detection for dead task recovery
        self.last_message_time: float = time.time()
        self._idle_check_task: Optional[asyncio.Task] = None
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
        is_tombstone = isinstance(orchestrator, DeadOrchestrator)

        if not self.save_success_only or orchestrator.is_success():
            # Run CPU-intensive work in thread pool
            start_time = time.perf_counter()
            loop = asyncio.get_event_loop()
            self.pending_writes += 1  # Track pending write before yielding
            try:
                data_to_write = await loop.run_in_executor(
                    None,
                    partial(
                        _write_output, await orchestrator.to_output(), self.output_path
                    ),
                )
                self.output_file.write(data_to_write)
                self.sink_write_latency.set(time.perf_counter() - start_time)  # type: ignore[attr-defined]
            finally:
                self.pending_writes -= 1  # Always decrement, even on error

        # Get registration order and remove from inflight tracking
        completed_order = self.inflight_orchestrators.pop(orchestrator.id, None)

        if is_tombstone:
            self.num_dead += 1
        else:
            self.e2e_latency.set(latency)  # type: ignore[attr-defined]
            latency = orchestrator.init_timestamp - orchestrator.creation_timestamp
            self.task_init_latency.set(latency)  # type: ignore[attr-defined]

            # Check if this orchestrator was in zombie set (came back from the dead)
            is_zombie_return = orchestrator.id in self.zombie_orchestrators
            if is_zombie_return:
                # Remove from zombie set - it actually completed
                self.zombie_orchestrators.discard(orchestrator.id)
                self.logger.info(
                    f"Zombie orchestrator {orchestrator.id} returned, removing from zombie set"
                )
                self.dead_order_window += self.LATE_ARRIVAL_INCR

            if not is_zombie_return:
                self.num_done += 1

            # Position-based zombie detection: when a non-error orchestrator completes,
            # all tasks registered more than max_concurrent_tasks before it should be done
            if (
                self.dead_orchestrator_tracking
                and not orchestrator.is_error()
                and completed_order is not None
            ):
                threshold = completed_order - self.dead_order_window
                if threshold > 0:
                    to_zombify = 0
                    # Iterate through oldest entries first, using lazy deletion
                    # (skip entries already removed from inflight_orchestrators)
                    while (
                        self.inflight_order
                        and self.inflight_order[0][0] <= threshold
                        and to_zombify < self.MAX_NEW_ZOMBIE
                    ):
                        order, orch_id = self.inflight_order.pop(0)
                        # Only zombify if still in inflight (not already completed)
                        if orch_id in self.inflight_orchestrators:
                            del self.inflight_orchestrators[orch_id]
                            self.zombie_orchestrators.add(orch_id)
                            to_zombify += 1
                            # Increase num_done to release semaphore
                            self.num_done += 1

                    if to_zombify:
                        self.logger.info(
                            f"Moved {to_zombify} orchestrators to zombie (threshold order: {threshold})"
                        )

            if self.metrics_accumulator:
                self.metrics_accumulator.accumulate(orchestrator)

        # Don't close output file if there are zombies or pending writes
        if (
            self.num_inputs is not None
            and self.num_done >= self.num_inputs
            and not self.zombie_orchestrators
            and self.pending_writes == 0
        ):
            self.output_file.close()

        return {"orchestrator": orchestrator}

    async def get_progress(self) -> int:
        # Cap num_done when there are zombies to prevent input pipeline from finishing early
        if self.zombie_orchestrators and self.num_inputs is not None:
            return min(self.num_done, self.num_inputs - 1)
        return self.num_done

    async def register_inflight(self, orchestrator_id: str):
        """Register an orchestrator as in-flight (before sending to first agent)."""
        self.registration_counter += 1
        order = self.registration_counter
        self.inflight_orchestrators[orchestrator_id] = order
        self.inflight_order.append((order, orchestrator_id))

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
        self.logger.debug(
            f"Creating tombstone for lost orchestrator: {orchestrator_id}"
        )

        # Create a DeadOrchestrator and process it through normal flow
        dead_orch = DeadOrchestrator(orchestrator_id)
        await self.receive_message(dead_orch)

    async def _idle_check_loop(self):
        """Background task to detect idle state and write tombstones for lost tasks."""
        while self.running:
            try:
                await asyncio.sleep(10.0)  # Check every 10 seconds

                # Skip if we're done and no zombies
                has_zombies = bool(self.zombie_orchestrators)
                if (
                    self.num_inputs is not None
                    and self.num_done >= self.num_inputs
                    and not has_zombies
                ):
                    break

                # Check if we've been idle long enough
                idle_time = time.time() - self.last_message_time
                if idle_time < self.IDLE_TIMEOUT:
                    continue

                # Check if all actors have empty queues and no pending tasks
                all_idle = await self._check_all_actors_idle()
                if not all_idle:
                    continue

                for orch_id in self.inflight_orchestrators:
                    self.zombie_orchestrators.add(orch_id)
                    self.num_done += 1
                self.inflight_orchestrators.clear()

                # All actors are idle - confirm zombies are dead and write tombstones
                if self.zombie_orchestrators:
                    self.logger.warning(
                        f"System idle for {idle_time:.1f}s with {len(self.zombie_orchestrators)} "
                        f"zombie tasks. Writing tombstones to confirm dead."
                    )
                    tasks = [
                        self._write_tombstone(orch_id)
                        for orch_id in self.zombie_orchestrators
                    ]
                    self.zombie_orchestrators.clear()
                    await asyncio.gather(*tasks)

                    # Wait for tombstones to be processed through the queue
                    while self.queue.qsize() > 0 or len(self.pending_tasks) > 0:
                        await asyncio.sleep(1)
                    # File will be closed by preprocess when last tombstone is written
                    break

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
            all_actors = [actor for actors in self._team.values() for actor in actors]

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

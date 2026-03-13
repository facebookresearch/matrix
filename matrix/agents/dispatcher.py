# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import time
from typing import Dict, Optional

import ray
from ray.util.metrics import Gauge

from .orchestrator import DeadOrchestrator, Orchestrator

logger = logging.getLogger(__name__)


class Dispatcher:
    """
    Dispatcher actor (one per role) that acts as a message broker.

    Agents pull work from their Dispatcher via checkout(), process it,
    and submit results back via submit(). The Dispatcher tracks exactly
    which agent has which orchestrator.

    Dead-agent detection is handled by agent restarts: when an agent
    restarts (max_restarts=-1), it calls agent_started() which tombstones
    any previously checked-out orchestrators for that agent.
    """

    def __init__(self, role: str, sink: ray.actor.ActorHandle, namespace: str):
        self.role = role
        self.sink = sink
        self.namespace = namespace

        self.incoming_queue: asyncio.Queue = asyncio.Queue()
        # orch_id -> (agent_ray_name, orchestrator)
        self.checked_out: Dict[str, tuple[str, Orchestrator]] = {}

        # Other role Dispatchers for forwarding (handles are stable — dispatchers don't restart)
        self.dispatchers: Dict[str, ray.actor.ActorHandle] = {}
        # Known agents (ray actor names), populated by agent_started()
        self.known_agents: set[str] = set()

        self.logger = logging.getLogger(f"Dispatcher[{role}]")

        self.queue_size = Gauge(
            "dispatcher_queue_size",
            description="Current queue size for this dispatcher",
            tag_keys=("role",),
        )
        self.queue_size.set_default_tags({"role": role})

    async def set_dispatchers(self, dispatchers: Dict[str, ray.actor.ActorHandle]):
        """Wire this Dispatcher to other role Dispatchers for forwarding."""
        self.dispatchers = dispatchers

    async def enqueue(self, orchestrator: Orchestrator):
        """Called by other Dispatchers or the framework to queue work."""
        orchestrator.enqueue_timestamp = time.time()
        await self.incoming_queue.put(orchestrator)
        self.queue_size.set(self.incoming_queue.qsize())

    async def checkout(self, agent_ray_name: str) -> Optional[Orchestrator]:
        """
        Agent pulls work, blocks until available.
        Returns None as shutdown sentinel.
        """
        orchestrator = await self.incoming_queue.get()
        self.queue_size.set(self.incoming_queue.qsize())
        if orchestrator is None:
            return None
        self.checked_out[orchestrator.id] = (agent_ray_name, orchestrator)
        return orchestrator

    async def submit(self, processed_orch: Orchestrator, orch_id: str):
        """
        Agent acks completion. Dispatcher removes from checked_out,
        checks is_done()/current_agent(), forwards to target Dispatcher or Sink.
        """
        self.checked_out.pop(orch_id, None)

        if await processed_orch.is_done():
            await self.sink.receive_message.remote(processed_orch)
        else:
            next_role = processed_orch.current_agent()
            if next_role == "_sink":
                await self.sink.receive_message.remote(processed_orch)
            elif next_role in self.dispatchers:
                await self.dispatchers[next_role].enqueue.remote(processed_orch)
            else:
                self.logger.error(
                    f"Unknown target role '{next_role}' for orch {orch_id}, sending to sink"
                )
                await self.sink.receive_message.remote(processed_orch)

    async def submit_error(self, orchestrator: Orchestrator, orch_id: str):
        """Agent error ack. Forward directly to Sink."""
        self.checked_out.pop(orch_id, None)
        await self.sink.receive_message.remote(orchestrator)

    async def agent_started(self, agent_ray_name: str):
        """
        Called on agent (re)start. Registers the agent and tombstones any
        previously checked-out orchestrators for it (handles restart race).
        """
        self.known_agents.add(agent_ray_name)
        to_tombstone = [
            (oid, orch)
            for oid, (aname, orch) in self.checked_out.items()
            if aname == agent_ray_name
        ]
        for oid, orch in to_tombstone:
            del self.checked_out[oid]
            self.logger.warning(
                f"Agent {agent_ray_name} restarted, tombstoning orch {oid}"
            )
            dead = DeadOrchestrator(
                oid, error=f"Agent {agent_ray_name} restarted while processing"
            )
            await self.sink.receive_message.remote(dead)

    async def shutdown(self):
        """Put None sentinels in queue (one per known agent) to stop their loops."""
        for _ in self.known_agents:
            await self.incoming_queue.put(None)

    async def check_health(self):
        return True

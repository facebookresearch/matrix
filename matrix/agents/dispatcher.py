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

from .agent_utils import remote_call_with_retry
from .orchestrator import DeadOrchestrator, Orchestrator

logger = logging.getLogger(__name__)


class Dispatcher:
    """
    Dispatcher actor (one per role) that acts as a message broker.

    Receives orchestrators via enqueue(), pushes them to agents via
    receive_message() round-robin, and routes completed work to the next
    Dispatcher or Sink.

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
        # Agent handles for push-based delivery, populated by set_agents()
        self.agents: Dict[str, ray.actor.ActorHandle] = {}
        self._agent_names: list[str] = []
        self._next_idx: int = 0

        self.logger = logging.getLogger(f"Dispatcher[{role}]")

        self.queue_size = Gauge(
            "dispatcher_queue_size",
            description="Current queue size for this dispatcher",
            tag_keys=("role",),
        )
        self.queue_size.set_default_tags({"role": role})

        # Event loop waits for first agent to register via agent_started()
        self._agents_ready = asyncio.Event()
        self._loop_task = asyncio.get_event_loop().create_task(self._event_loop())

    async def set_dispatchers(self, dispatchers: Dict[str, ray.actor.ActorHandle]):
        """Wire this Dispatcher to other role Dispatchers for forwarding."""
        self.dispatchers = dispatchers

    async def enqueue(self, orchestrator: Orchestrator):
        """Called by other Dispatchers or the framework to queue work."""
        orchestrator.enqueue_timestamp = time.time()
        await self.incoming_queue.put(orchestrator)
        self.queue_size.set(self.incoming_queue.qsize())

    async def _event_loop(self):
        """Dequeue orchestrators and push to agents round-robin."""
        await self._agents_ready.wait()
        while True:
            orchestrator = await self.incoming_queue.get()
            if orchestrator is None:  # shutdown sentinel
                break
            self.queue_size.set(self.incoming_queue.qsize())
            await self._push_to_agent(orchestrator)

    async def _push_to_agent(self, orchestrator: Orchestrator):
        """Push orchestrator to an agent with round-robin and retry."""
        if not self._agent_names:
            self.logger.error(f"No agents registered for role {self.role}")
            dead = DeadOrchestrator(
                orchestrator.id, error=f"No agents for role {self.role}"
            )
            await self._send_to_sink(dead)
            return

        num_agents = len(self._agent_names)
        for attempt in range(num_agents):
            idx = (self._next_idx + attempt) % num_agents
            agent_name = self._agent_names[idx]
            agent_handle = self.agents[agent_name]
            try:
                await remote_call_with_retry(
                    agent_handle.receive_message,
                    orchestrator,
                    logger=self.logger,
                )
                self._next_idx = (idx + 1) % num_agents
                self.checked_out[orchestrator.id] = (agent_name, orchestrator)
                return
            except Exception as e:
                self.logger.warning(
                    f"Failed to push to agent {agent_name}: {repr(e)}"
                )
                continue

        # All agents failed
        self.logger.error(
            f"All agents unreachable for orch {orchestrator.id}, tombstoning"
        )
        dead = DeadOrchestrator(
            orchestrator.id, error=f"All agents for {self.role} unreachable"
        )
        await self._send_to_sink(dead)

    async def submit(self, processed_orch: Orchestrator, orch_id: str):
        """
        Agent acks completion. Dispatcher removes from checked_out,
        checks is_done()/current_agent(), forwards to target Dispatcher or Sink.
        """
        self.checked_out.pop(orch_id, None)

        if await processed_orch.is_done():
            await self._send_to_sink(processed_orch)
        else:
            next_role = processed_orch.current_agent()
            if next_role == "_sink":
                await self._send_to_sink(processed_orch)
            elif next_role in self.dispatchers:
                try:
                    await remote_call_with_retry(
                        self.dispatchers[next_role].enqueue,
                        processed_orch,
                        logger=self.logger,
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to forward orch {orch_id} to dispatcher {next_role}: {repr(e)}"
                    )
                    dead = DeadOrchestrator(
                        orch_id,
                        error=f"Failed to forward to {next_role}: {e}",
                    )
                    await self._send_to_sink(dead)
            else:
                self.logger.error(
                    f"Unknown target role '{next_role}' for orch {orch_id}, sending to sink"
                )
                await self._send_to_sink(processed_orch)

    async def submit_error(self, orchestrator: Orchestrator, orch_id: str):
        """Agent error ack. Forward directly to Sink."""
        self.checked_out.pop(orch_id, None)
        await self._send_to_sink(orchestrator)

    async def agent_started(self, agent_ray_name: str, agent_handle: ray.actor.ActorHandle):
        """
        Called on agent (re)start. Updates the agent handle and tombstones
        any previously checked-out orchestrators for this agent.
        """
        # Register / update agent handle
        self.agents[agent_ray_name] = agent_handle
        if agent_ray_name not in self._agent_names:
            self._agent_names.append(agent_ray_name)
        if not self._agents_ready.is_set():
            self._agents_ready.set()

        # Atomically remove all checked-out entries for this agent (no await
        # between removals, so _push_to_agent cannot interleave here)
        to_tombstone = []
        for oid in list(self.checked_out):
            aname, orch = self.checked_out[oid]
            if aname == agent_ray_name:
                del self.checked_out[oid]
                to_tombstone.append((oid, orch))

        # Now send tombstones — _push_to_agent may interleave at these await
        # points, but any new checked_out entries are for the new agent instance
        for oid, orch in to_tombstone:
            self.logger.warning(
                f"Agent {agent_ray_name} restarted, tombstoning orch {oid}"
            )
            dead = DeadOrchestrator(
                oid, error=f"Agent {agent_ray_name} restarted while processing"
            )
            await self._send_to_sink(dead)

    async def _send_to_sink(self, orchestrator: Orchestrator):
        """Send to sink with retry. If sink is unreachable, log and drop."""
        try:
            await remote_call_with_retry(
                self.sink.receive_message, orchestrator, logger=self.logger
            )
        except Exception as e:
            self.logger.error(
                f"Failed to send orch {orchestrator.id} to sink after retries: {repr(e)}"
            )

    async def shutdown(self):
        """Stop event loop and signal all agents to shut down."""
        # Stop the event loop
        await self.incoming_queue.put(None)
        # Signal each agent to shut down
        for agent_name, agent_handle in self.agents.items():
            try:
                await agent_handle.shutdown.remote()
            except Exception as e:
                self.logger.warning(
                    f"Error shutting down agent {agent_name}: {repr(e)}"
                )

    async def check_health(self):
        return True

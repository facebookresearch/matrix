# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Tau2-bench orchestrator using LangGraph routing.

Same interaction and side-effect logic as Tau2Orchestrator in tau2_bench.py,
but the routing is expressed as a LangGraph StateGraph.

Because ``update`` does more than determine the next agent (it calls
``sync_tools`` and ``should_stop`` which are async / have side effects),
the method is overridden here: it pre-computes state, lets the graph
decide the next node, then runs the side effects.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Type, TypedDict

from langgraph.graph import END, StateGraph
from omegaconf import DictConfig

from matrix.agents.langgraph_orchestrator import LangGraphOrchestrator
from matrix.agents.p2p_agents import (
    AgentActor,
    BaseResourceClient,
    Orchestrator,
    Sink,
)

from .tau2_bench_utils import (
    build_simulation_output,
    check_should_stop,
    run_initialization_actions,
    sync_tools,
)

# -- State schema: what the routing functions see --------------------------


class Tau2State(TypedDict):
    should_stop: bool
    has_tool_calls: bool
    last_non_env_agent: str  # for remote_env → caller routing


# -- Graph -----------------------------------------------------------------


def build_tau2_graph():
    """
    Build the Tau2 routing graph.

    Flow::

        START → user_simulator ⇄ llm_agent
                     ↓                ↓
                  remote_env ←--------+  (on tool_calls)
                     ↓
                (back to caller)

        any node ──(should_stop)──→ remote_reward → END
    """
    graph = StateGraph(Tau2State)

    for name in ("user_simulator", "llm_agent", "remote_env", "remote_reward"):
        graph.add_node(name, lambda s: s)

    graph.set_entry_point("user_simulator")

    graph.add_conditional_edges(
        "user_simulator",
        lambda s: (
            "remote_reward"
            if s["should_stop"]
            else "remote_env" if s["has_tool_calls"] else "llm_agent"
        ),
    )

    graph.add_conditional_edges(
        "llm_agent",
        lambda s: (
            "remote_reward"
            if s["should_stop"]
            else "remote_env" if s["has_tool_calls"] else "user_simulator"
        ),
    )

    graph.add_conditional_edges(
        "remote_env",
        lambda s: ("remote_reward" if s["should_stop"] else s["last_non_env_agent"]),
    )

    graph.add_edge("remote_reward", END)

    return graph.compile()


# -- Orchestrator ----------------------------------------------------------


class Tau2LangGraphOrchestrator(LangGraphOrchestrator):
    """
    Drop-in replacement for Tau2Orchestrator using LangGraph routing.

    Shared helpers (``sync_tools``, ``check_should_stop``, ``build_simulation_output``,
    ``run_initialization_actions``) live in ``tau2_bench_utils`` — only the routing
    decision in ``update`` is delegated to the graph.
    """

    def __init__(self, step_limit: int):
        self.step_limit = step_limit
        self.user_system_message: Optional[Dict[str, Any]] = None
        self.need_sync = False
        super().__init__(
            graph=build_tau2_graph(),
            # state_reducer unused — update is fully overridden
            state_reducer=lambda h, t: {},
            entry_point="user_simulator",
        )

    # -- lifecycle ----------------------------------------------------------

    async def init(
        self,
        simulation_id: str,
        first_agent: Tuple[Type, DictConfig],
        sink: Sink,
        metadata: dict[str, Any],
        resources: dict[str, BaseResourceClient],
        logger: logging.Logger,
    ):
        task = metadata["task"]
        self._id = task["id"]
        await super().init(
            simulation_id, first_agent, sink, metadata, resources, logger
        )
        needs_sync = await run_initialization_actions(
            task["initial_state"],
            resources["container"],
            self.resource_state["container"],
            logger,
        )
        if needs_sync:
            await sync_tools(
                resources["container"], self.resource_state["container"], logger
            )

    # -- update: graph routing + side effects -------------------------------

    async def update(
        self,
        result: Any,
        updater: AgentActor,
        logger: logging.Logger,
    ) -> Orchestrator:
        if logger is not None:
            logger.debug(
                f"Orchestrator {self.id} updating from {self._current_node} "
                f"with result {result}"
            )

        # 1. Append to history
        await Orchestrator.update(self, result, updater, logger)
        resources = updater.resources

        # 2. Mark sync needed when leaving remote_env
        if self._current_node == "remote_env":
            self.need_sync = True

        # 3. Pre-compute async state for routing
        stop, reason = await check_should_stop(self.history, self.step_limit)
        if reason:
            self.status["termination_reason"] = reason
        last_non_env = next(
            (m.agent for m in reversed(self.history) if m.agent != "remote_env"),
            "user_simulator",
        )
        state = Tau2State(
            should_stop=stop,
            has_tool_calls="tool_calls" in self.history[-1].response,
            last_non_env_agent=last_non_env,
        )

        # 4. Let the graph decide the next node
        self._current_node = self._get_next_node(self._current_node, state)

        # 5. Sync tools when transitioning back to an LLM agent after env call
        if self.need_sync and self._current_node in ("llm_agent", "user_simulator"):
            await sync_tools(
                resources["container"], self.resource_state["container"], logger
            )
            self.need_sync = False

        return self

    # -- output conversion -------------------------------------------------

    async def to_simulation(self) -> Dict[str, Any]:
        return await build_simulation_output(
            self.history, self.id, self.status.get("termination_reason", "unknown")
        )

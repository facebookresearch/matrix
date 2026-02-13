# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
LangGraph-compatible orchestrator for the Matrix P2P Agent Framework.

Allows users to define agent routing with a LangGraph StateGraph.
Nodes in the graph correspond to Matrix agent names; after each agent
processes, the graph's edges determine the next agent.

State is derived from Matrix's history on every transition (history is
the single source of truth).

Usage:

    from langgraph.graph import StateGraph, END

    class MyState(TypedDict):
        agreement: bool

    graph = StateGraph(MyState)
    graph.add_node("teacher", lambda s: s)   # placeholder — Matrix agent does the work
    graph.add_node("student", lambda s: s)
    graph.set_entry_point("teacher")
    graph.add_edge("teacher", "student")
    graph.add_conditional_edges(
        "student",
        lambda s: END if s["agreement"] else "teacher",
    )

    orchestrator = LangGraphOrchestrator(
        graph=graph.compile(),
        state_reducer=my_reducer,   # (history, task) -> MyState
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Tuple, Type

from omegaconf import DictConfig

from .p2p_agents import AgentActor, BaseResourceClient, HistPair, Orchestrator, Sink

# Sentinels — also re-exported so users don't need langgraph just for these.
END = "__end__"
START = "__start__"

logger = logging.getLogger(__name__)


class LangGraphOrchestrator(Orchestrator):
    """
    Orchestrator whose update function is defined by a LangGraph StateGraph.

    Graph nodes are identity placeholders; Matrix AgentActors do the real
    processing. The graph only controls *routing*: after an agent finishes,
    the compiled graph's edges (including conditional edges) decide the
    next agent.

    State is *derived* from ``self.history`` via the ``state_reducer``
    callable on every transition — there is no separate LangGraph state
    object to keep in sync.
    """

    def __init__(
        self,
        graph: Any,  # langgraph CompiledGraph
        state_reducer: Callable[[List[HistPair], dict | None], dict],
        entry_point: str | None = None,
    ):
        super().__init__()
        self._graph = graph
        self._state_reducer = state_reducer
        self._current_node: str = entry_point or self._resolve_entry_point(graph)
        self._task: dict | None = None

    # ------------------------------------------------------------------
    # Orchestrator interface
    # ------------------------------------------------------------------

    def current_agent(self) -> str:
        if self._current_node in (END, "__end__"):
            raise ValueError("Graph has reached END, no current agent")
        return self._current_node

    async def init(
        self,
        simulation_id: str,
        first_agent: Tuple[Type, DictConfig],
        sink: Sink,
        metadata: dict[str, Any],
        resources: dict[str, BaseResourceClient],
        logger: logging.Logger,
    ) -> None:
        self._task = metadata.get("task")
        await super().init(
            simulation_id, first_agent, sink, metadata, resources, logger
        )

    async def update(
        self,
        result: Any,
        updater: AgentActor,
        logger: logging.Logger,
    ) -> Orchestrator:
        await super().update(result, updater, logger)
        state = self._state_reducer(self.history, self._task)
        self._current_node = self._get_next_node(self._current_node, state)
        return self

    async def is_done(self) -> bool:
        if not self.history:
            return False
        if not self.history[-1].response.get("status_ok", True):
            return True
        return self._current_node in (END, "__end__")

    # ------------------------------------------------------------------
    # Graph traversal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_entry_point(graph: Any) -> str:
        """Extract the entry-point node from a compiled LangGraph graph."""
        try:
            inner = graph.graph if hasattr(graph, "graph") else graph
            if hasattr(inner, "edges"):
                for source, target in inner.edges:
                    if source in (START, "__start__"):
                        return target
            if hasattr(graph, "builder") and hasattr(graph.builder, "_entry_point"):
                return graph.builder._entry_point
        except Exception:
            pass
        raise ValueError(
            "Could not determine entry point from graph. "
            "Please provide entry_point explicitly."
        )

    def _get_next_node(self, current_node: str, state: dict) -> str:
        """Walk the compiled graph's edges / branches to find the next node."""
        inner = (
            self._graph.graph
            if hasattr(self._graph, "graph")
            else self._graph
        )

        # 1. Unconditional edges
        if hasattr(inner, "edges"):
            for source, target in inner.edges:
                if source == current_node:
                    return target

        # 2. Conditional edges (branches)
        if hasattr(inner, "branches"):
            branches = inner.branches
            if current_node in branches:
                for _name, branch in branches[current_node].items():
                    if hasattr(branch, "path"):
                        key = branch.path(state)
                        if hasattr(branch, "ends") and branch.ends:
                            return branch.ends.get(key, key)
                        return key

        raise ValueError(
            f"No edge from node '{current_node}' in the graph. "
            f"State: {state}"
        )

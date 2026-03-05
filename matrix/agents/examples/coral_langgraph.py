# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Coral teacher-student interaction using LangGraph routing.

Same interaction pattern as CoralOrchestrator in ts_interaction.py,
but the routing logic is expressed as a LangGraph StateGraph.
"""

import logging
from typing import List, TypedDict

from langgraph.graph import END, StateGraph

from matrix.agents.langgraph_orchestrator import LangGraphOrchestrator
from matrix.agents.p2p_agents import HistPair

from .ts_interaction import CoralTaskMixin

# -- State schema: what the routing functions see --------------------------


class CoralState(TypedDict):
    agreement: bool
    agreement_correctness: bool
    rated_turns: int
    has_error: bool


# -- Graph definition ------------------------------------------------------


def build_coral_graph(max_turns: int = 20):
    """
    Build the Coral routing graph.
    """
    graph = StateGraph(CoralState)

    # Nodes are identity — Matrix agents do the real work
    for name in ("teacher", "student", "answer_extractor", "answer_matcher"):
        graph.add_node(name, lambda s: s)

    graph.set_entry_point("teacher")
    graph.add_edge("teacher", "answer_extractor")
    graph.add_edge("student", "answer_extractor")
    graph.add_edge("answer_extractor", "answer_matcher")
    graph.add_conditional_edges(
        "answer_matcher",
        lambda s: (
            END
            if s["agreement"] or s["rated_turns"] >= max_turns
            else "student" if s["rated_turns"] % 2 == 1 else "teacher"
        ),
    )

    return graph.compile()


# -- State reducer: history -> CoralState ----------------------------------


def coral_state_reducer(
    history: List[HistPair], task: dict | None = None
) -> CoralState:
    if not history:
        return CoralState(
            agreement=False,
            agreement_correctness=False,
            rated_turns=0,
            has_error=False,
        )
    last = history[-1].response
    return CoralState(
        agreement=last.get("agreement", False),
        agreement_correctness=last.get("agreement_correctness", False),
        rated_turns=last.get("rated_turns", 0),
        has_error=not last.get("status_ok", True),
    )


# -- Orchestrator ----------------------------------------------------------


class CoralLangGraphOrchestrator(CoralTaskMixin, LangGraphOrchestrator):
    """
    Drop-in replacement for CoralOrchestrator using LangGraph routing.
    """

    def __init__(self, max_turns: int):
        super().__init__(
            graph=build_coral_graph(max_turns),
            state_reducer=coral_state_reducer,  # type: ignore[arg-type]
            entry_point="teacher",
        )
        self.max_turns = max_turns

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
from typing import Any, Dict, List, Tuple, Type, TypedDict

from langgraph.graph import END, StateGraph
from omegaconf import DictConfig

from matrix.agents.langgraph_orchestrator import LangGraphOrchestrator
from matrix.agents.p2p_agents import BaseResourceClient, HistPair, Sink

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

    Flow:
        teacher -> answer_extractor -> answer_matcher --(agreement/error/max_turns)--> END
                       ^                     |
                       |                     v
                       +---- student <-------+
    """
    graph = StateGraph(CoralState)

    # Nodes are identity — Matrix agents do the real work
    for name in ("teacher", "student", "answer_extractor", "answer_matcher"):
        graph.add_node(name, lambda s: s)

    graph.set_entry_point("teacher")
    graph.add_edge("teacher", "answer_extractor")
    graph.add_edge("answer_extractor", "answer_matcher")
    graph.add_conditional_edges(
        "answer_matcher",
        lambda s: (
            END
            if s["has_error"] or s["agreement"] or s["rated_turns"] >= max_turns
            else "student"
        ),
    )
    graph.add_edge("student", "teacher")

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


class CoralLangGraphOrchestrator(LangGraphOrchestrator):
    """
    Drop-in replacement for CoralOrchestrator using LangGraph routing.

    The only extra code vs. the base class is Coral-specific task setup
    (question_id, answer, options) that the agents read from the
    orchestrator — identical to what CoralOrchestrator already does.
    """

    def __init__(self, max_turns: int = 20):
        super().__init__(
            graph=build_coral_graph(max_turns),
            state_reducer=coral_state_reducer,
            entry_point="teacher",
        )

    async def init(
        self,
        simulation_id: str,
        first_agent: Tuple[Type, DictConfig],
        sink: Sink,
        metadata: dict[str, Any],
        resources: dict[str, BaseResourceClient],
        logger: logging.Logger,
    ) -> None:
        task = metadata["task"]
        self._id = task.get("question_id", task.get("id"))
        await super().init(
            simulation_id, first_agent, sink, metadata, resources, logger
        )
        # Fields that CoralAgent / CoralExtractionAgent read off the orchestrator
        self.task_answer = task.get("answer")
        self.task_options = "\n".join(
            f"({k}) {v}" for k, v in task.get("choices", {}).items()
        )

    async def is_done(self) -> bool:
        done = await super().is_done()
        if done and self.history:
            self.status["success"] = self.history[-1].response.get(
                "agreement_correctness", False
            )
        return done

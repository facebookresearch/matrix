# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import MISSING, asdict, dataclass, fields
from io import StringIO
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, TypeVar

import ray
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf

from matrix import Cli
from matrix.utils.ray import get_ray_address

from ..agent_utils import extract_json
from ..p2p_agents import (
    AgentActor,
    BaseMetricsAccumulator,
    BaseResourceClient,
    Orchestrator,
    Sink,
)
from ..p2p_extended import (
    HistPair,
    HuggingfaceDatasetLoader,
    LLMAgentActor,
    SequentialOrchestrator,
)


@dataclass
class CriteriaScores:
    criteria_1: float
    criteria_2: float
    criteria_3: float
    criteria_4: float


@dataclass
class ExamQuestionResult:
    exam_question: str
    question_category: str
    extracted_answer: str
    independent_answer: List[str]


# ==== Coral Simulation State ====
class NROrchestrator(SequentialOrchestrator):

    async def init(
        self,
        simulation_id: str,
        first_agent: Tuple[Type, DictConfig],
        sink: "Sink",
        metadata: dict[str, Any],
        resources: dict[str, "BaseResourceClient"],
        logger: logging.Logger,
    ) -> None:
        task = metadata["task"]
        self._id = task["metadata"]["WARC-Record-ID"]
        await super().init(
            simulation_id, first_agent, sink, metadata, resources, logger
        )

    async def is_done(self) -> bool:
        return (
            self.current_agent == "_sink"
            or self.status.get("success") == False
            or not self.history[-1].response.get("status_ok")
        )

    async def update(
        self,
        result: Any,
        updater: "AgentActor",
        logger: logging.Logger,
    ) -> "Orchestrator":
        """Update the state with the agent's result and determine the next agent."""
        if logger is not None:
            logger.debug(
                f"Orchestrator {self.id} updating from {self.current_agent()} with result {result}"
            )
        await super().update(result, updater, logger)
        termination_reason = result.get("termination_reason")
        if termination_reason:
            self.status["termination_reason"] = termination_reason
            self.status["success"] = result.get("success")
        return self


# ==== Concrete Agent Implementation (Example) ====
@ray.remote
class FilterAgent(LLMAgentActor):

    async def preprocess(self, orchestrator: Orchestrator) -> List[Dict[str, str]]:  # type: ignore[override]
        task = await orchestrator.get_task()
        is_en = task.get("language_id_whole_page_fasttext", {"en": 0}).get(
            "en", 0
        ) >= self.config.get("filter_en", 0)
        if not is_en:
            return None  # type: ignore[return-value]
        else:
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": task["text"]},
            ]

    async def process(self, orchestrator: Orchestrator, response: Any) -> Any:
        if response is None:
            return None
        return await super().process(orchestrator, response)

    async def postprocess(self, orchestrator: Orchestrator, response: Any) -> Any:  # type: ignore[override]
        if response is not None:
            response = await super().postprocess(orchestrator, response)
        return FilterAgent.process_response(response)

    @staticmethod
    def process_response(response):
        if response is None:
            response = {
                "status_ok": True,
                "success": False,
                "termination_reason": "filter_by_en",
            }
        else:
            if response.get("text") != "Yes":
                response |= {
                    "status_ok": True,
                    "success": False,
                    "termination_reason": "filter_by_classifier",
                }
        return response

    @classmethod
    async def get_task_message(
        cls, agent_config: DictConfig, task: Dict[str, Any]
    ) -> Dict[str, Any]:
        return None  # type: ignore[return-value]


@ray.remote
class ScoreAgent(LLMAgentActor):

    async def preprocess(self, orchestrator: Orchestrator) -> List[Dict[str, str]]:  # type: ignore[override]
        task = await orchestrator.get_task()
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task["text"]},
        ]

    async def postprocess(self, orchestrator: Orchestrator, response: Any) -> Any:  # type: ignore[override]
        response = await super().postprocess(orchestrator, response)
        return ScoreAgent.process_response(response, self.logger)

    @staticmethod
    def process_response(response, logger):
        if response["status_ok"]:
            try:
                data = extract_json(response["text"], CriteriaScores)
                if not all(v > 0 for v in vars(data).values()):
                    response |= {
                        "success": False,
                        "termination_reason": "filter_by_score",
                    }

            except Exception as e:
                msg = f"Failed to extract json {repr(e)}"
                logger.error(msg)
                response |= {
                    "success": False,
                    "termination_reason": "filter_by_score_wo_json",
                    "error": msg,
                }
        return response


@ray.remote
class QuestionAgent(LLMAgentActor):

    async def preprocess(self, orchestrator: Orchestrator) -> List[Dict[str, str]]:  # type: ignore[override]
        task = await orchestrator.get_task()
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task["text"]},
        ]

    async def postprocess(self, orchestrator: Orchestrator, response: Any) -> Any:  # type: ignore[override]
        response = await super().postprocess(orchestrator, response)
        return QuestionAgent.process_response(response, self.logger)

    @staticmethod
    def process_response(response, logger):
        if response["status_ok"]:
            try:
                data: ExamQuestionResult = extract_json(response["text"], ExamQuestionResult)  # type: ignore[assignment]
                if not data.exam_question:
                    response |= {
                        "success": False,
                        "termination_reason": "filter_by_no_question",
                    }
                elif not data.independent_answer:
                    response |= {
                        "success": False,
                        "termination_reason": "filter_by_no_answer",
                    }
                elif "\\boxed{" not in data.independent_answer[-1]:
                    response |= {
                        "success": False,
                        "termination_reason": "filter_by_no_boxed_answer",
                    }
                else:
                    response |= {
                        "success": True,
                        "termination_reason": "success",
                        **asdict(data),
                    }

            except Exception as e:
                msg = f"Failed to extract json {repr(e)}"
                logger.error(msg)
                response |= {
                    "success": False,
                    "termination_reason": "filter_by_question_wo_json",
                    "error": msg,
                }
        return response


class NRMetricsAccumulator(BaseMetricsAccumulator):
    def __init__(self):
        super().__init__()

        self.overall_metrics = defaultdict(int)

    def accumulate(self, orchestrator: NROrchestrator):  # type: ignore[override]
        self.overall_metrics["total_input"] += 1
        reason = orchestrator.status.get("termination_reason")
        if reason:
            self.overall_metrics[reason] += 1
        last_turn = orchestrator.history[-1].response if orchestrator.history else {}
        self.overall_metrics["conv_err"] += not last_turn.get("status_ok", False)

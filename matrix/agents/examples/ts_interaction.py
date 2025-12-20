# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import os
import re
from collections import defaultdict
from io import StringIO
from typing import Any, Dict, Generator, List, Optional, Tuple, Type

import ray
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf

from matrix import Cli
from matrix.utils.ray import get_ray_address

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


# ==== Coral Simulation State ====
class CoralOrchestrator(SequentialOrchestrator):
    def __init__(
        self,
        interaction_order: List[str],
        max_turns: int,
    ):
        super().__init__(
            interaction_order,
        )
        self.max_turns = max_turns

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
        self._id = task["question_id"]
        await super().init(
            simulation_id, first_agent, sink, metadata, resources, logger
        )
        # copy of metadata
        self.task_answer = task.get("answer", None)
        self.task_options = "\n".join(
            [f"({letter}) {option}" for letter, option in task["choices"].items()]
        )

    async def is_done(self) -> bool:
        return (
            not self.history[-1].response.get("status_ok", True)
            or self.history[-1].response.get("agreement", False)
            or self.history[-1].response.get("rated_turns", 0) >= self.max_turns
        )


# ==== Concrete Agent Implementation (Example) ====
@ray.remote
class CoralAgent(LLMAgentActor):

    async def preprocess(self, orchestrator: CoralOrchestrator) -> List[Dict[str, str]]:  # type: ignore[override]
        messages = [
            await msg.response.get_async("text")
            for msg in orchestrator.history
            if msg.agent in {"student", "teacher"}
        ]
        if len(messages) % 2 == 0:
            messages.insert(0, "hi")  # Changed from prepend to insert
        return [{"role": "system", "content": self.system_prompt}] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": msg}
            for i, msg in enumerate(messages)
        ]

    @classmethod
    async def get_task_message(
        cls, agent_config: DictConfig, task: Dict[str, Any]
    ) -> Dict[str, Any]:
        init_student_prompt = (
            f"I'm trying to solve this problem: \"{task['question']}\"\n"
            + "And the choices are:\n"
            + "\n".join(
                [f"({letter}) {option}" for letter, option in task["choices"].items()]
            )
        )
        initial_message = {
            "agent": "student",
            "response": {"text": init_student_prompt},
        }
        return initial_message


@ray.remote
class CoralExtractionAgent(LLMAgentActor):

    async def preprocess(self, orchestrator: CoralOrchestrator) -> List[Dict[str, str]]:  # type: ignore[override]
        response = await orchestrator.history[-1].response.get_async("text")
        task = await orchestrator.get_task()
        with StringIO() as prompt_builder:
            prompt_builder.write(
                f"**This is the original question:**\n{task['question']}\n\n"
                + f"\n\n**And the options are:**\n{orchestrator.task_options}\n\n"
                + "**This is the response you need to extract answer from:**\n\n"
            )
            prompt_builder.write(response)
            prompt_builder.write(
                '\n\n**Extract the answer in ([A-Z]) format, or say "not sure yet".**'
            )
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt_builder.getvalue()},
            ]

    def normalize_answer(self, response: str) -> tuple[str, str | None]:
        if "not sure yet" in response.lower():
            return "not sure yet", None

        pattern = r"\(([A-Z])\)"
        matches = re.findall(pattern, response)

        if len(matches) > 0:
            answer = matches[0]
            return f"The answer is {answer}.", answer

        return f"Error: {response}", None

    async def postprocess(self, orchestrator: CoralOrchestrator, response: Any) -> Any:  # type: ignore[override]
        if "text" in response and isinstance(response["text"], list):
            response["text"] = response["text"][0]
        response["status_ok"] = "error" not in response
        answer_msg, answer = self.normalize_answer(response.get("text", ""))
        response["extracted_answer"] = answer
        response["valid_answer"] = "The answer is " in answer_msg
        response["correct"] = answer is not None and answer == orchestrator.task_answer
        return response


@ray.remote
class CoralMatchAgent(AgentActor):

    async def preprocess(self, orchestrator: CoralOrchestrator) -> Dict[str, Any]:  # type: ignore[override]
        response: Dict[str, Any] = {"status_ok": True}
        belief_dict = {}
        reversed = orchestrator.history[::-1]
        for index, turn in enumerate(reversed):
            if "extracted_answer" in turn.response and turn.response.get(
                "valid_answer", False
            ):
                role = reversed[index + 1].agent if index + 1 < len(reversed) else None
                assert role in {"student", "teacher"}
                if role not in belief_dict:
                    belief_dict[role] = turn.response["extracted_answer"]
        response["matched_answer"] = belief_dict
        if len(belief_dict) == 2 and belief_dict.get("student") == belief_dict.get(
            "teacher"
        ):
            response["agreement"] = True
        else:
            response["agreement"] = False
        response["agreement_correctness"] = (
            response["agreement"] and orchestrator.task_answer in belief_dict.values()
        )
        response["rated_turns"] = (
            len(
                [
                    msg
                    for msg in orchestrator.history[::-1]
                    if msg.agent == self.agent_id
                ]
            )
            + 1
        )
        return response


class CoralDatasetLoader(HuggingfaceDatasetLoader):
    def transform(self, item):
        item = dict(item)
        choices = {chr(65 + i): option for i, option in enumerate(item["options"])}
        item["choices"] = choices
        return item


# ==== Coral Metrics Accumulator ====
class CoralMetricsAccumulator(BaseMetricsAccumulator):
    def get_avg_ts_len(self, result_conv: List[HistPair]) -> Tuple[float, float, int]:
        assert result_conv[0].agent in [
            "student",
            "teacher",
        ], "Unsupported role! Only student and teacher roles are supported right now!"

        t_lens = []
        s_lens = []
        for turn in result_conv:
            response = turn.response
            if turn.agent == "student" and response.get("status_ok", False):
                s_lens.append(response.get("usage", {}).get("completion_tokens", 0))
            elif turn.agent == "teacher" and response.get("status_ok", False):
                t_lens.append(response.get("usage", {}).get("completion_tokens", 0))

        avg_t_len = (sum(t_lens) / len(t_lens)) if len(t_lens) > 0 else 0
        avg_s_len = (sum(s_lens) / len(s_lens)) if len(s_lens) > 0 else 0
        total_conv_len = sum(t_lens) + sum(s_lens)

        return avg_t_len, avg_s_len, total_conv_len

    def accumulate(self, orchestrator: CoralOrchestrator):  # type: ignore[override]
        last_turn = orchestrator.history[-1].response if orchestrator.history else {}
        self.overall_metrics["conv_err"].append(not last_turn.get("status_ok", False))
        self.overall_metrics["agreement"].append(last_turn.get("agreement", False))
        self.overall_metrics["agreement_correctness"].append(
            last_turn.get("agreement_correctness", False)
        )

        conv_list = [
            msg for msg in orchestrator.history if msg.agent in {"student", "teacher"}
        ]
        conv_list = conv_list[1:]  # skip the injected prompt
        self.overall_metrics["total_turns"].append(len(conv_list))

        if conv_list:
            t_len, s_len, total_len = self.get_avg_ts_len(conv_list)
            self.overall_metrics["t_len"].append(t_len)
            self.overall_metrics["s_len"].append(s_len)
            self.overall_metrics["conv_len"].append(total_len)

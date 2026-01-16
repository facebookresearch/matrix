# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Dummy agents for measuring system overhead without actual LLM calls.
These agents generate random responses to simulate realistic payloads.
"""

import random
import string
import time
from typing import Any, Dict, List

import ray
from omegaconf import DictConfig

from ..p2p_agents import AgentActor
from .ts_interaction import CoralOrchestrator


# Small base text block (~900 chars)
_BASE_TEXT = (
    "To find the amount of trade discount, we need to subtract the net price from the list price. "
    "List price equals four dollars and ninety five cents. Net price equals two dollars and ninety five cents. "
    "Trade discount equals list price minus net price. Therefore the trade discount is two dollars. "
    "So the amount of trade discount is two dollars. The correct answer is option I. "
    "Let me verify this calculation by working through the problem step by step. "
    "First we identify the given values in the problem. The list price is the original price before any discounts. "
    "The net price is what the customer actually pays after the discount is applied. "
    "To find the discount amount we simply subtract the net price from the list price. "
    "This gives us the total savings or discount that was applied to the purchase. "
    "In this case the calculation shows that the customer saved two dollars on this transaction. "
)
_BASE_LEN = len(_BASE_TEXT)


def generate_random_text(length: int = 1024) -> str:
    """Return text of specified length. Repeats base text if needed."""
    if length <= _BASE_LEN:
        return _BASE_TEXT[:length]
    # Repeat base text enough times to cover length
    repeats = (length // _BASE_LEN) + 1
    return (_BASE_TEXT * repeats)[:length]


@ray.remote
class DummyCoralAgent(AgentActor):
    """
    Dummy version of CoralAgent that generates random text responses.
    Used for measuring system overhead without actual LLM calls.

    Config options:
        generation_length: Target text length in chars (default: 1024)
        length_delta: Variation as decimal, e.g. 0.2 means 80%-120% (default: 0.2)
    """

    async def preprocess(self, orchestrator: CoralOrchestrator) -> Dict[str, Any]:
        # Get config params with defaults
        base_length = self.config.get("generation_length", 1024)
        delta = self.config.get("length_delta", 0.2)

        # Calculate actual length with random variation
        min_len = int(base_length * (1 - delta))
        max_len = int(base_length * (1 + delta))
        length = random.randint(min_len, max_len)

        text = generate_random_text(length)

        # Calculate realistic token counts (roughly 4 chars per token)
        prompt_tokens = random.randint(200, 500)
        completion_tokens = len(text) // 4

        return {
            "finish_reason": ["stop"],
            "response_timestamp": time.time(),
            "text": text,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "status_ok": True,
        }

    async def process(self, orchestrator: CoralOrchestrator, response: Any) -> Any:
        # No actual LLM call, just return the preprocessed response
        return response

    async def postprocess(self, orchestrator: CoralOrchestrator, response: Any) -> Any:
        response["timestamp"] = time.time()
        return response

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
class DummyCoralExtractionAgent(AgentActor):
    """
    Dummy version of CoralExtractionAgent that randomly extracts an answer.
    Used for measuring system overhead without actual LLM calls.
    """

    async def preprocess(self, orchestrator: CoralOrchestrator) -> Dict[str, Any]:
        # Randomly select an answer from available options
        options = list(orchestrator.task_options.split("\n"))
        if options:
            # Extract just the letter from format "(A) option text"
            available_letters = []
            for opt in options:
                if opt.startswith("(") and len(opt) > 2:
                    available_letters.append(opt[1])

            if available_letters:
                answer = random.choice(available_letters)
            else:
                answer = random.choice(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
        else:
            answer = random.choice(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])

        text = f"({answer})"

        # Simulate realistic token usage for extraction
        prompt_tokens = random.randint(300, 400)
        completion_tokens = random.randint(2, 5)

        return {
            "finish_reason": ["stop"],
            "response_timestamp": time.time(),
            "text": text,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "status_ok": True,
            "extracted_answer": answer,
            "valid_answer": True,
            "correct": answer == orchestrator.task_answer,
        }

    async def process(self, orchestrator: CoralOrchestrator, response: Any) -> Any:
        # No actual LLM call, just return the preprocessed response
        return response

    async def postprocess(self, orchestrator: CoralOrchestrator, response: Any) -> Any:
        response["timestamp"] = time.time()
        return response


# Pre-computed answer options to avoid repeated list creation
_ANSWER_OPTIONS = ("A", "B", "C", "D", "E")


@ray.remote
class DummyCoralMatchAgent(AgentActor):
    """
    Dummy version of CoralMatchAgent that randomly determines agreement.
    Used for measuring system overhead without actual LLM calls.

    Config options:
        expected_turns: Target number of turns before agreement (default: 3)
    """

    async def preprocess(self, orchestrator: CoralOrchestrator) -> Dict[str, Any]:
        # Count rated_turns with simple iteration (no list creation)
        rated_turns = 1
        for msg in orchestrator.history:
            if msg.agent == self.agent_id:
                rated_turns += 1

        # Get expected turns from config
        expected_turns = self.config.get("expected_turns", 3)

        # Agreement logic: low prob before expected_turns, high prob at/after
        if rated_turns < expected_turns:
            # 10% chance of early agreement
            agreement = random.randint(1, 10) == 1
        else:
            # 90% chance of agreement once we reach expected turns
            agreement = random.randint(1, 10) <= 9

        # Generate a single random answer for simplicity
        answer = _ANSWER_OPTIONS[random.randint(0, 4)]
        belief_dict = {"student": answer, "teacher": answer}

        if not agreement:
            # Make answers different
            other = _ANSWER_OPTIONS[(random.randint(0, 4) + 1) % 5]
            belief_dict["teacher"] = other

        return {
            "status_ok": True,
            "matched_answer": belief_dict,
            "agreement": agreement,
            "agreement_correctness": agreement and orchestrator.task_answer == answer,
            "rated_turns": rated_turns,
        }

    async def process(self, orchestrator: CoralOrchestrator, response: Any) -> Any:
        # No actual processing needed, just return the preprocessed response
        return response

    async def postprocess(self, orchestrator: CoralOrchestrator, response: Any) -> Any:
        response["timestamp"] = time.time()
        return response

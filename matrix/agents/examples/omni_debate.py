# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List

import ray
from omegaconf import DictConfig

from ..agent_actor import LLMAgentActor
from ..dataset_loader import HuggingfaceDatasetLoader
from ..orchestrator import SequentialOrchestrator
from ..p2p_agents import BaseMetricsAccumulator

logger = logging.getLogger(__name__)


# ==== Dataset ====
class DebateDatasetLoader(HuggingfaceDatasetLoader):
    """Load debate topics from ibm-research/debate_speeches, deduplicated by topic_id."""

    def transform(self, item):
        item = dict(item)
        return {"topic_id": item["topic_id"], "topic": item["topic"]}

    def load_data(self):
        seen_topics = set()
        count = 0
        # Stream all rows (set parent cut_off high), dedup here with our own cut_off
        original_cut_off = self.cut_off
        self.cut_off = None  # let parent stream everything
        for item in super().load_data():
            tid = item["topic_id"]
            if tid not in seen_topics:
                seen_topics.add(tid)
                count += 1
                yield item
                if original_cut_off and count >= original_cut_off:
                    break
        self.cut_off = original_cut_off
        self._count = count
        self.done = True


# ==== Orchestrator ====
class DebateOrchestrator(SequentialOrchestrator):
    def __init__(self, interaction_order: List[str], max_turns: int = 3):
        super().__init__(interaction_order)
        self.max_turns = max_turns

    async def init(
        self,
        simulation_id: str,
        first_agent: "Any",
        sink: "Any",
        metadata: dict[str, Any],
        resources: dict[str, Any],
        logger: logging.Logger,
    ) -> None:
        task = metadata["task"]
        self._id = str(task["topic_id"])
        await super().init(simulation_id, first_agent, sink, metadata, resources, logger)

    async def is_done(self) -> bool:
        if not self.history:
            return False
        last = self.history[-1].response
        if not last.get("status_ok", True):
            return True
        # Count actual debate turns (exclude the seed at index 0)
        turns = sum(
            1 for msg in self.history[1:]
            if msg.agent in {"proponent", "opponent"}
        )
        # Each round = 2 turns (proponent + opponent), max_turns = number of rounds
        done = turns >= self.max_turns * 2
        if done:
            self.status["success"] = True
        return done


# ==== Debate Agent ====
@ray.remote
class DebateAgent(LLMAgentActor):

    async def preprocess(self, orchestrator: DebateOrchestrator) -> List[Dict[str, str]]:
        """Build messages for the LLM. Only send text history, not audio."""
        task = await orchestrator.get_task()
        topic = task["topic"]

        # Collect text-only messages from debate history, skip the seed (index 0)
        debate_msgs = []
        for msg in orchestrator.history[1:]:
            if msg.agent not in {"proponent", "opponent"}:
                continue
            text = await msg.response.get_async("text")
            if text:
                debate_msgs.append((msg.agent, text))

        # Build chat messages: system prompt + topic + alternating user/assistant
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.append({
            "role": "user",
            "content": f'The debate topic is: "{topic}"\n\nPlease present your opening argument.',
        })

        # Append debate history as alternating assistant/user turns
        for agent, text in debate_msgs:
            role = "assistant" if agent == self.agent_id else "user"
            messages.append({"role": role, "content": text})

        # Ensure the last message is from "user" so the model can respond
        if messages[-1]["role"] == "assistant":
            messages.append({
                "role": "user",
                "content": "Please continue with your next argument.",
            })

        return messages

    async def postprocess(self, orchestrator: DebateOrchestrator, response: Any) -> Any:
        """Store both text and audio in history."""
        return await super().postprocess(orchestrator, response)

    @classmethod
    async def get_task_message(
        cls, agent_config: DictConfig, task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Seed the debate with the topic."""
        return {
            "agent": "proponent",
            "response": {
                "text": f'Debate topic: "{task["topic"]}"',
            },
        }


# ==== Metrics ====
class DebateMetricsAccumulator(BaseMetricsAccumulator):
    def accumulate(self, orchestrator: DebateOrchestrator):
        self.overall_metrics["conv_err"].append(orchestrator.is_error())
        debate_turns = [
            msg for msg in orchestrator.history
            if msg.agent in {"proponent", "opponent"}
        ]
        self.overall_metrics["total_turns"].append(len(debate_turns))

        # Track avg response lengths
        for role in ("proponent", "opponent"):
            lens = [
                msg.response.get("usage", {}).get("completion_tokens", 0)
                for msg in debate_turns
                if msg.agent == role and msg.response.get("status_ok", False)
            ]
            avg_len = (sum(lens) / len(lens)) if lens else 0
            self.overall_metrics[f"{role}_avg_tokens"].append(avg_len)

        # Track how many turns had audio
        audio_count = sum(
            1 for msg in debate_turns if msg.response.get("audio")
        )
        self.overall_metrics["audio_turns"].append(audio_count)

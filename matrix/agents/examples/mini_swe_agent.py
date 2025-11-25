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
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import ray
from omegaconf import DictConfig, OmegaConf

from matrix import Cli
from matrix.utils.os import run_async
from matrix.utils.ray import get_ray_address

from ..agent_utils import render_template, setup_logging
from ..p2p_agents import (
    AgentActor,
    BaseMetricsAccumulator,
    BaseResourceClient,
    Orchestrator,
    Sink,
)
from ..p2p_extended import (
    ContainerExecutionAgent,
    ContainerResourceClient,
    HistoryOrchestrator,
    HuggingfaceDatasetLoader,
    LLMAgentActor,
)

"""
error handling:
1. if can't obtain container, done on this item during initialization.
2. if execution failed, we can continue feedback to llm.
3. if llm error, done on this item.
"""


def get_swebench_docker_image_name(instance: dict, docker_cache: str | None) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        image_name = "docker://" + image_name
    if image_name.startswith("docker://") and docker_cache is not None:
        file_name = os.path.join(
            docker_cache, os.path.basename(image_name).replace(":", "_") + ".sif"
        )
        if os.path.exists(file_name):
            image_name = file_name
    return image_name


class SweContainerClient(ContainerResourceClient):
    def __init__(
        self,
        resource_id,
        matrix_service: str,
        matrix_cli,
        docker_cache: str | None,
        start_config: Optional[Dict[str, Any]] = None,
        exec_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            resource_id,
            matrix_service,
            matrix_cli,
            start_config=start_config,
            exec_params=exec_params,
        )
        self.docker_cache = os.path.expanduser(docker_cache) if docker_cache else None

    def get_container_image(self, task):
        image_name = get_swebench_docker_image_name(task, self.docker_cache)
        return image_name


# ==== Swe Simulation State ====
class SweOrchestrator(HistoryOrchestrator):
    def __init__(self, step_limit: int):
        super().__init__()
        self._current_agent = "coder"
        self.step_limit = step_limit

    async def init(
        self,
        simulation_id: str,
        first_agent: Tuple[Type, DictConfig],
        sink: "Sink",
        metadata: dict[str, Any],
        resources: dict[str, "BaseResourceClient"],
        logger: logging.Logger,
    ):
        task = metadata["task"]
        self._id = task["instance_id"]
        await super().init(
            simulation_id, first_agent, sink, metadata, resources, logger
        )

    def current_agent(self) -> str:
        return self._current_agent

    async def is_done(self) -> bool:
        num_llm_calls = len([msg for msg in self.history if msg.agent == "coder"])
        if num_llm_calls > self.step_limit:
            self.status["exceeded_step_limit"] = True
            return True
        if self.history[-1].agent == "executor":
            text = await self.history[-1].response.get_async("output", "")
            lines = text.lstrip().splitlines()
            success = len(lines) > 0 and lines[0].strip() in [
                "MINI_SWE_AGENT_FINAL_OUTPUT",
                "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            ]
            self.status["success"] = success
            if success:
                self.status["diff"] = "\n".join(lines[1:])
            return success
        if self.history[-1].agent == "coder" and not self.history[-1].response.get(
            "status_ok", False
        ):
            # can't recover from llm error
            return True
        return False

    async def update(
        self,
        result: Any,
        updater: "AgentActor",
        logger: logging.Logger,
    ) -> "Orchestrator":
        """Update the state with the agent's result and determine the next agent."""
        await super().update(result, updater, logger)
        match self._current_agent:
            case "coder":
                self._current_agent = "reviewer"
            case "reviewer":
                self._current_agent = (
                    "executor" if "action" in self.history[-1].response else "coder"
                )
            case "executor":
                self._current_agent = "coder"
        return self


# ==== Concrete Agent Implementation (Example) ====
@ray.remote
class SweCodeAgent(LLMAgentActor):

    async def preprocess(self, orchestrator: SweOrchestrator) -> List[Dict[str, str]]:  # type: ignore[override]
        messages = [
            await msg.response.get_async("text")
            for msg in orchestrator.history
            if msg.agent in {"coder", "executor"}
            or (msg.agent == "reviewer" and "action" not in msg.response)
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
        text = render_template(
            agent_config.instance_template, task=task["problem_statement"]
        )
        initial_message = {"agent": "coder", "response": {"text": text}}
        return initial_message


@ray.remote
class SweExecutionAgent(ContainerExecutionAgent):

    async def get_commands(
        self, orchestrator: HistoryOrchestrator
    ) -> List[Union[str, List[str]]]:
        cmd = orchestrator.history[-1].response["action"]
        return [cmd]

    async def postprocess(self, orchestrator: HistoryOrchestrator, results: Any) -> Any:  # type: ignore[override]
        result = results["results"][0]
        response: Dict[str, Any] = {"status_ok": True}
        returncode = result["returncode"]
        output = result["output"]
        response["output"] = output
        if returncode != 0 and ("TimeoutError" in output or "TimeoutExpired" in output):
            cmd = orchestrator.history[-1].response["action"]
            observation = render_template(
                self.config.timeout_template, output=output, action={"action": cmd}
            )
            response["text"] = observation
        else:
            observation = render_template(
                self.config.action_observation_template, output=result
            )
            response["text"] = observation

        return response


@ray.remote
class SweReviewAgent(AgentActor):

    async def preprocess(self, orchestrator: SweOrchestrator) -> Dict[str, Any]:  # type: ignore[override]
        response: Dict[str, Any] = {"status_ok": True}
        text = (await orchestrator.history[-1].response.get_async("text")) or ""
        actions = re.findall(r"```bash\n(.*?)\n```", text, re.DOTALL)
        if len(actions) == 1:
            response["action"] = actions[0].strip()
        else:
            response["status_ok"] = False
            response["text"] = render_template(
                self.config.format_error_template, actions=actions
            )
        return response


class SweDatasetLoader(HuggingfaceDatasetLoader):
    def __init__(
        self,
        name: str,
        split: str,
        dataset_mapping: Dict[str, str],
        cut_off: Optional[int] = None,
    ):
        super().__init__(dataset_mapping[name], split, cut_off=cut_off)


# ==== Swe Metrics Accumulator ====
class SweMetricsAccumulator(BaseMetricsAccumulator):
    def __init__(self, pred_file: str):
        super().__init__()
        self.pred_file = os.path.abspath(pred_file)
        self.output_data: dict[str, Any] = {}

    def accumulate(self, orchestrator: SweOrchestrator):  # type: ignore[override]
        last_turn = orchestrator.history[-1].response if orchestrator.history else {}
        self.overall_metrics["conv_err"].append(not last_turn.get("status_ok", False))

        conv_list = [msg for msg in orchestrator.history if msg.agent in {"coder"}]
        conv_list = conv_list[1:]  # skip the injected prompt
        self.overall_metrics["total_llm"].append(len(conv_list))

        conv_list = [msg for msg in orchestrator.history if msg.agent in {"executor"}]
        self.overall_metrics["total_execution"].append(len(conv_list))

        self.overall_metrics["exceeded_step_limit"].append(
            orchestrator.status.get("exceeded_step_limit", False)
        )
        self.overall_metrics["success"].append(
            orchestrator.status.get("success", False)
        )

        instance_id = orchestrator.id

        if orchestrator.status.get("success", False):
            assert orchestrator.history[-1].agent == "executor"
            result = orchestrator.status.get("diff", "")
        else:
            result = ""  # json.dumps(orchestrator.status)
        self.output_data[instance_id] = {
            "model_name_or_path": "custom_model",
            "instance_id": instance_id,
            "model_patch": result,
        }

    def done(self):
        with open(self.pred_file, "w") as f:
            json.dump(self.output_data, f)
        return super().done()

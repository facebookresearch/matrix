# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import logging
import os
import pprint
import re
import tempfile
from collections import defaultdict
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple, Type

import ray
import yaml
from jinja2 import Template
from omegaconf import DictConfig, OmegaConf

from matrix import Cli
from matrix.client.container_client import ContainerClient
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
    LLMAgentActor,
    LLMResourceClient,
)

"""
error handling:
1. if can't obtain container, done on this item during initialization.
2. if execution failed, we can continue feedback to llm.
3. if llm error, done on this item.
"""

logger = logging.getLogger(__name__)


def pprint_agent(instructions: Any) -> str:
    buffer = StringIO()
    pprint.pprint(instructions, stream=buffer, width=80)
    formatted_str = buffer.getvalue()
    return formatted_str


# retry connection error to allow tau2 server to start inside container
CURL_COMMAND_PREFIX = [
    "curl",
    "-s",
    "--fail-with-body",
    "--retry",
    "50",
    "--retry-delay",
    "3",
    "--retry-connrefused",
]
CURL_POST_PREFIX = CURL_COMMAND_PREFIX + [
    "-X",
    "POST",
    "-H",
    "Content-Type: application/json",
    "-d",
]
CURL_POST_PREFIX_VERBOSE = CURL_COMMAND_PREFIX + [
    "-v",
    "-X",
    "POST",
    "-H",
    "Content-Type: application/json",
    "-d",
]


def query_tau2_server(
    resource_client: "BaseResourceClient", endpoint: str, logger
) -> str:
    async def helper():
        resource_info = await resource_client.acquire({}, logger)
        try:
            result = await resource_client.utilize(
                resource_info,
                logger,
                cmd=CURL_COMMAND_PREFIX + [f"http://localhost:8004/{endpoint}"],
            )
            logger.debug(
                f"query_tau2_server {endpoint} got {result} with {resource_info}"
            )
            if result["returncode"] != 0:
                # okay to raise, failed for resource and agent init
                raise RuntimeError(f"Failed to query {endpoint} for {result}")
            policy = result.get("output", "").strip()
            return policy
        finally:
            await resource_client.release(resource_info, logger)

    return run_async(helper())


# ==== Tau2 Simulation State ====
STOP = "###STOP###"
TRANSFER = "###TRANSFER###"
OUT_OF_SCOPE = "###OUT-OF-SCOPE###"


class Tau2Orchestrator(HistoryOrchestrator):
    def __init__(self, step_limit: int):
        super().__init__()
        self._current_agent = "user_simulator"
        self.step_limit = step_limit
        self.user_system_message: Optional[Dict[str, Any]] = None
        self.need_sync = False

    def current_agent(self) -> str:
        return self._current_agent

    async def sync_tools(self, resources: dict[str, "BaseResourceClient"], logger):
        container_client = resources["container"]
        result = await container_client.utilize(
            self.resource_state["container"],
            logger,
            cmd=CURL_POST_PREFIX + [{}, "http://localhost:8004/api/v1/sync_tools"],
        )
        if "returncode" not in result or result["returncode"] != 0:
            logger.error(
                f"Failed to sync_tools with {result}, rewards will catch the issue"
            )

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
        self._id = task["id"]
        await super().init(
            simulation_id, first_agent, sink, metadata, resources, logger
        )
        initial_state = task["initial_state"]
        if initial_state:
            container_client = resources["container"]
            assert (
                initial_state.get("initialization_data") is None
            ), "initiailization_data not supported yet"
            assert (
                initial_state.get("message_history") is None
            ), "message_history not supported yet"
            actions = initial_state.get("initialization_actions", [])
            for call in actions:
                cmd = CURL_POST_PREFIX + [
                    call,
                    f"http://localhost:8004/api/v1/run_env_function",
                ]
                result = await container_client.utilize(
                    self.resource_state["container"],
                    logger,
                    cmd=cmd,
                )
                if logger is not None:
                    logger.debug(f"initial action {cmd}, result {result}")
                if (
                    "returncode" not in result
                    or result["returncode"] != 0
                    or "Not Found" in result.get("output", "")
                ):
                    # fail the task during initialization
                    raise Exception(result)
            if actions:
                await self.sync_tools(resources, logger)

    async def is_done(self) -> bool:
        return self._current_agent is None

    async def should_stop(self) -> bool:
        num_llm_calls = len(
            [
                msg
                for msg in self.history
                if msg.agent in ["user_simulator", "llm_agent"]
            ]
        )
        if num_llm_calls > self.step_limit:
            self.status["termination_reason"] = "max_steps"
            return True
        if self.history[-1].agent in [
            "user_simulator",
            "llm_agent",
        ] and not self.history[-1].response.get("status_ok", False):
            self.status["termination_reason"] = "too_many_errors"
            # can't recover from llm error
            return True

        if self.history[-1].agent == "user_simulator":
            text = await self.history[-1].response.get_async("text", "")
            is_stop = STOP in text or TRANSFER in text or OUT_OF_SCOPE in text
            if is_stop:
                self.status["termination_reason"] = "user_stop"

            return is_stop
        return False

    async def update(
        self,
        result: Any,
        updater: "AgentActor",
        logger: logging.Logger,
    ) -> "Orchestrator":
        """Update the state with the agent's result and determine the next agent."""
        if logger is not None:
            logger.debug(
                f"Orchestrator {self.id} updating from {self._current_agent} with result {result}"
            )
        await super().update(result, updater, logger)
        resources = updater.resources
        if self._current_agent == "remote_reward":
            self._current_agent = None  # type: ignore[assignment]
            return self
        if await self.should_stop():
            self._current_agent = "remote_reward"
            return self
        is_tool_call = self.history and "tool_calls" in self.history[-1].response
        if is_tool_call:
            self._current_agent = "remote_env"
        elif self._current_agent == "user_simulator":
            self._current_agent = "llm_agent"
            if self.need_sync:
                await self.sync_tools(resources, logger)
                self.need_sync = False
        elif self._current_agent == "llm_agent":
            self._current_agent = "user_simulator"
            if self.need_sync:
                await self.sync_tools(resources, logger)
                self.need_sync = False
        elif self._current_agent == "remote_env":
            # find the last non env call backwards from self.history
            last = None
            for msg in reversed(self.history):
                if msg.agent != "remote_env":
                    last = msg.agent
                    break
            assert last is not None
            self._current_agent = last
            self.need_sync = True
        return self

    async def to_simulation(self) -> Dict[str, Any]:
        messages = []
        last_role = None
        for turn_idx, msg in enumerate(self.history):
            role = (
                "tool"
                if msg.agent == "remote_env"
                else "user" if msg.agent == "user_simulator" else "assistant"
            )
            msg_type = (
                "ToolMessage"
                if role == "tool"
                else "UseMessage" if role == "user" else "AssistantMessage"
            )
            # convert to src/tau2/data_model/simulation
            message = {
                "type": msg_type,
                "role": role,
                "content": await msg.response.get_async("text"),
                "turn_idx": turn_idx,
            }
            tool_calls = msg.response.get("tool_calls")
            if tool_calls:
                tool_calls_fixed = []
                for call in tool_calls:
                    arguments = call["function"]["arguments"]
                    try:
                        data = json.loads(arguments)
                    except Exception:
                        # hack for weak models
                        data = {}
                    tool_calls_fixed.append(
                        {
                            "id": call["id"],
                            "name": call["function"]["name"],
                            "arguments": data,
                            "requestor": role,
                        }
                    )
                message["tool_calls"] = tool_calls_fixed
            if role == "tool":
                message |= {
                    "id": msg.response["tool_call_id"],
                    "requestor": last_role,
                    "error": not msg.response["status_ok"],
                }
            if role != "tool":
                last_role = role
            messages.append(message)
        return {
            "id": self.id,
            "task_id": self.id,
            "termination_reason": self.status["termination_reason"],
            "messages": messages,
            "start_time": "",
            "end_time": "",
            "duration": 1,
        }


# user_simulator: user for llm_agent;  assistant for user_simulator
# llm_agent: assistant for llm_agent; user for user_simulator
# remote_env's next agent is the caller, the content is the json.dumps of the response
# once no more tools call, pass to the next agent, skip the tools calls.
# @ray.remote


class Tau2LLMResourceClient(LLMResourceClient):
    async def init(self, resources: dict[Any, BaseResourceClient], logger):
        container = resources["container"]
        tools = json.loads(query_tau2_server(container, "api/v1/get_tools", logger))
        self.tools = list(
            tools[
                "tools" if self.resource_id == "llm_agent_llm" else "user_tools"
            ].values()
        )
        if len(self.tools) > 0:
            logger.debug(f"tools {self.tools}")
            self.tools_params |= {
                "tools": self.tools,
                "tool_choice": "auto",
            }


class Tau2LLMAgent(LLMAgentActor):
    def __init__(
        self,
        id: str,
        agent_id: str,
        config: DictConfig,
        resources: dict[str, BaseResourceClient],
    ):
        super().__init__(
            id,
            agent_id,
            config,
            resources=resources,
        )
        if agent_id == "llm_agent":
            # instantiate the template
            container_client = resources["container"]
            policy = query_tau2_server(
                container_client, "api/v1/get_policy", self.logger
            )
            self.system_prompt = render_template(
                self.system_prompt, domain_policy=policy
            )
        else:
            llm_client = resources["user_simulator_llm"]
            if llm_client.tools:  # type: ignore[attr-defined]
                self.system_prompt = config["system_prompt_with_tools"]

    async def get_system_message(
        self, orchestrator: Tau2Orchestrator
    ) -> Dict[str, Any]:
        return {"role": "system", "content": self.system_prompt}

    async def preprocess(self, orchestrator: Tau2Orchestrator) -> List[Dict[str, str]]:  # type: ignore[override]
        messages = []
        for msg in orchestrator.history:
            # skip unrelated tool messages
            if (
                msg.agent == "remote_env" and msg.response["caller"] != self.agent_id
            ) or ("tool_calls" in msg.response and msg.agent != self.agent_id):
                continue
            to_add = {}
            if self.agent_id == "llm_agent":
                role = (
                    "user"
                    if msg.agent == "user_simulator"
                    else "assistant" if msg.agent == "llm_agent" else "tool"
                )
            else:
                role = (
                    "user"
                    if msg.agent == "llm_agent"
                    else "assistant" if msg.agent == "user_simulator" else "tool"
                )
            text = await msg.response.get_async("text")
            if text:
                to_add["content"] = text
            if msg.response.get("tool_calls"):
                to_add["tool_calls"] = msg.response["tool_calls"]
            if msg.response.get("tool_call_id"):
                to_add["tool_call_id"] = msg.response["tool_call_id"]
            messages.append({"role": role} | to_add)
        return [await self.get_system_message(orchestrator)] + messages

    async def postprocess(self, orchestrator: Orchestrator, response: Any) -> Any:  # type: ignore[override]
        if (
            "tool_calls" not in response
            and "text" in response
            and isinstance(response["text"], list)
        ):
            # just a message to the other party
            # fix gpt-oss output
            try:
                message = json.loads(response["text"][0])
                if isinstance(message, dict):
                    if len(message) == 2 and "role" in message and "content" in message:
                        response["text"][0] = message["content"]
                    elif len(message) == 1 and "message" in message:
                        response["text"][0] = message["message"]
            except:
                pass
        return await super().postprocess(orchestrator, response)


@ray.remote
class Tau2UserSimulatorAgent(Tau2LLMAgent):

    async def get_system_message(
        self, orchestrator: Tau2Orchestrator
    ) -> Dict[str, Any]:
        """Get the system prompt for the agent."""
        if orchestrator.user_system_message is None:
            orchestrator.user_system_message = {
                "role": "system",
                "content": render_template(
                    self.system_prompt,
                    instructions=yaml.dump(
                        (await orchestrator.get_task())["user_scenario"],
                        sort_keys=False,
                        default_flow_style=False,
                    ),
                ),
            }

        return orchestrator.user_system_message

    @classmethod
    async def get_task_message(
        cls, agent_config: DictConfig, task: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "agent": "llm_agent",
            "response": {"text": "Hi! How can I help you today?", "status_ok": True},
        }


@ray.remote
class Tau2EnvAgent(ContainerExecutionAgent):

    async def get_commands(self, orchestrator):
        tool_callls = orchestrator.history[-1].response["tool_calls"]
        env_type = (
            "user"
            if orchestrator.history[-1].agent == "user_simulator"
            else "assistant"
        )
        commands = []
        for call in tool_callls:
            try:
                arguments = call["function"]["arguments"]
                data = json.loads(arguments)
            except Exception as e:
                self.logger.error(
                    f"Failed to parse json from tool call argument {arguments}"
                )
                # hack for weak models
                data = {}
            cmd = CURL_POST_PREFIX + [
                {
                    "env_type": env_type,
                    "func_name": call["function"]["name"],
                    "arguments": data,
                },
                f"http://localhost:8004/api/v1/run_tools",
            ]
            commands.append(cmd)
        return commands

    async def postprocess(self, orchestrator: Tau2Orchestrator, response: Any) -> Any:  # type: ignore[override]
        # todo deal with errors
        results = response["results"]
        response = []
        tool_calls = orchestrator.history[-1].response["tool_calls"]
        assert len(tool_calls) == len(results)
        for i, result in enumerate(results):
            text = result["output"]
            status_ok = result["returncode"] == 0
            if not status_ok and "detail" in text:
                try:
                    error = json.loads(text)
                    text = "Error: " + error["detail"]
                except Exception as e:
                    pass
            msg = {
                "tool_call_id": tool_calls[i]["id"],  # A unique ID for the tool call
                "text": text,
                "status_ok": status_ok,
                "caller": orchestrator.history[-1].agent,
            }
            response.append(msg)
        return response


@ray.remote
class Tau2RewardAgent(ContainerExecutionAgent):
    def __init__(
        self,
        id: str,
        agent_id: str,
        config: DictConfig,
        resources: dict[str, BaseResourceClient],
    ):
        super().__init__(
            id,
            agent_id,
            config,
            resources=resources,
        )
        self.tmp_dir = os.path.abspath(os.path.expanduser(config["tmp_dir"]))

    async def get_commands(self, orchestrator):
        data = {
            "task": await orchestrator.get_task(),
            "simulation": await orchestrator.to_simulation(),
            "evaluation_type": "all",
        }

        # Use with, but keep file (delete=False)
        with tempfile.NamedTemporaryFile(
            dir=self.tmp_dir, delete=False, suffix=".json", mode="w", encoding="utf-8"
        ) as tmp:
            json.dump(data, tmp, indent=2)
            json_path = tmp.name

        self.logger.debug(f"dump data to {json_path} for reward")
        cmd = CURL_POST_PREFIX + [
            f"@{json_path}",
            "http://localhost:8004/api/v1/get_reward",
        ]

        return [cmd]

    async def postprocess(self, orchestrator: Orchestrator, response: Any) -> Any:
        # todo deal with errors
        result = response["results"][0]
        response = {"status_ok": result["returncode"] == 0}
        try:
            reward_info = json.loads(result["output"])
            response["reward_info"] = reward_info
        except Exception as e:
            response["error"] = repr(e)

        return response


class Tau2MetricsAccumulator(BaseMetricsAccumulator):
    def __init__(self):
        super().__init__()

    def accumulate(self, orchestrator: Tau2Orchestrator):  # type: ignore[override]
        last_turn = orchestrator.history[-1].response if orchestrator.history else {}
        self.overall_metrics["conv_err"].append(not last_turn.get("status_ok", False))
        if "reward_info" in last_turn:
            self.overall_metrics["rewards"].append(
                last_turn["reward_info"].get("reward", 0)
            )
        tr = orchestrator.status.get("termination_reason", "unknown_error")
        for reason in ["user_stop", "max_steps", "too_many_errors", "unknown_error"]:
            self.overall_metrics[f"termination_{reason}"].append(
                1 if reason == tr else 0
            )

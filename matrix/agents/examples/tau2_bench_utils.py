# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared constants and helper functions for Tau2 orchestrators.
"""

import json
import pprint
from io import StringIO
from typing import Any, Optional, Tuple

from matrix.utils.os import run_async

# ==== Tau2 Simulation State ====
STOP = "###STOP###"
TRANSFER = "###TRANSFER###"
OUT_OF_SCOPE = "###OUT-OF-SCOPE###"

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


def pprint_agent(instructions: Any) -> str:
    buffer = StringIO()
    pprint.pprint(instructions, stream=buffer, width=80)
    formatted_str = buffer.getvalue()
    return formatted_str


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


async def sync_tools(container_client, container_resource_state, logger):
    """Sync tools on the Tau2 container."""
    result = await container_client.utilize(
        container_resource_state,
        logger,
        cmd=CURL_POST_PREFIX + [{}, "http://localhost:8004/api/v1/sync_tools"],
    )
    if "returncode" not in result or result["returncode"] != 0:
        logger.error(
            f"Failed to sync_tools with {result}, rewards will catch the issue"
        )


async def check_should_stop(history, step_limit) -> Tuple[bool, Optional[str]]:
    """
    Check whether the simulation should stop.

    Returns (should_stop, termination_reason).
    """
    num_llm_calls = len(
        [m for m in history if m.agent in ("user_simulator", "llm_agent")]
    )
    if num_llm_calls > step_limit:
        return True, "max_steps"
    if history[-1].agent in (
        "user_simulator",
        "llm_agent",
    ) and not history[
        -1
    ].response.get("status_ok", False):
        return True, "too_many_errors"
    if history[-1].agent == "user_simulator":
        text = await history[-1].response.get_async("text", "")
        is_stop = STOP in text or TRANSFER in text or OUT_OF_SCOPE in text
        if is_stop:
            return True, "user_stop"
    return False, None


async def build_simulation_output(history, sim_id, termination_reason) -> dict:
    """Build the simulation output dict from conversation history."""
    messages = []
    last_role = None
    for turn_idx, msg in enumerate(history):
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
        "id": sim_id,
        "task_id": sim_id,
        "termination_reason": termination_reason,
        "messages": messages,
        "start_time": "",
        "end_time": "",
        "duration": 1,
    }


async def run_initialization_actions(
    initial_state, container_client, container_resource_state, logger
) -> bool:
    """
    Run initialization actions for a Tau2 task.

    Returns True if actions were run (meaning sync_tools should be called).
    """
    if not initial_state:
        return False
    assert (
        initial_state.get("initialization_data") is None
    ), "initialization_data not supported yet"
    assert (
        initial_state.get("message_history") is None
    ), "message_history not supported yet"
    actions = initial_state.get("initialization_actions", [])
    for call in actions:
        cmd = CURL_POST_PREFIX + [
            call,
            "http://localhost:8004/api/v1/run_env_function",
        ]
        result = await container_client.utilize(
            container_resource_state,
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
            raise Exception(result)
    return bool(actions)

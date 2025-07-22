# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional

import packaging
from fastapi import FastAPI, HTTPException
from google import genai
from packaging import version
from ray import serve
from starlette.requests import Request
from vllm.entrypoints.openai.protocol import ChatCompletionRequest

logger = logging.getLogger("ray.serve")

app = FastAPI()


def _extract_version(name: str) -> version.Version | None:
    match = re.search(r"gemini-(\d+\.\d+)", name)
    return version.parse(match.group(1)) if match else None


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 1,
        "target_ongoing_requests": 64,
    },
    max_ongoing_requests=64,  # make this large so that multi-turn can route to the same replica
)
@serve.ingress(app)
class GeminiDeployment:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        thinking_budget: int,
    ):
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)
        self.thinking_budget = thinking_budget
        version = _extract_version(model_name)
        self.reasoning = version is not None and version >= packaging.version.parse(
            "2.5"
        )

    def _transform_messages(
        self, messages: List[Dict[str, str]]
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Transform OpenAI-style messages to the format recognized by Google Gemini API.
        Separates the system instruction and maps chat history.
        """
        system_instruction_content: Optional[str] = None
        transformed_contents: List[Dict[str, Any]] = []

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                if system_instruction_content is None:
                    system_instruction_content = content
                else:
                    # If multiple system messages, concatenate (or choose a different strategy)
                    system_instruction_content += "\n" + content
            elif role == "user":
                transformed_contents.append(
                    {
                        "role": "user",
                        "parts": [{"text": content}],
                    }
                )
            elif role == "assistant":
                transformed_contents.append(
                    {
                        "role": "model",
                        "parts": [{"text": content}],
                    }
                )
            else:
                logger.warning(
                    f"Unknown role '{role}' encountered. Skipping this message."
                )
        if transformed_contents and transformed_contents[0]["role"] == "model":
            logger.warning(
                "Transformed chat history starts with 'model' role, which Gemini's generate_content might reject. Consider adjusting the input history."
            )
        return transformed_contents, system_instruction_content

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        logger.debug(f"Request: {request}")
        completion_request = request.model_dump(exclude_unset=True)

        # gemini api expects only one message as input
        messages_transformed, system_instruction_content = self._transform_messages(
            completion_request.get("messages", [])
        )

        request_params: Dict[str, Any] = {
            "contents": messages_transformed,
            "config": {
                "temperature": completion_request.get("temperature", 0.6),
                "top_p": completion_request.get("top_p", 0.9),
                "seed": completion_request.get("seed", 42),
                "max_output_tokens": completion_request.get("max_tokens", 1024),
                "response_logprobs": completion_request.get("logprobs", False),
                "candidate_count": completion_request.get("n", 1),
                "system_instruction": system_instruction_content,
            },
        }
        if self.reasoning:
            request_params["config"]["thinking_config"] = {
                "thinking_budget": self.thinking_budget
            }
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name, **request_params
            )
        except genai.errors.APIError as e:
            raise HTTPException(status_code=e.code, detail=str(e))

        if response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
            }
        else:
            usage = {
                "prompt_tokens": 0,
                "total_tokens": 0,
                "completion_tokens": 0,
            }

        completion_response: Dict[str, Any] = {
            "id": response.response_id,
            "usage": usage,
        }
        if response.candidates is None:
            error: Dict[str, Any] = {}
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                error = {  # Adding an error field for clarity, not standard OpenAI format
                    "message": f"Content blocked due to: {response.prompt_feedback.block_reason.name}",
                    "type": "content_filter_error",
                    "code": response.prompt_feedback.block_reason.value,
                }
            else:
                error = {
                    "message": "Gemini API returned an error or no valid response candidates.",
                    "type": "api_error",
                    "code": None,
                }
            raise HTTPException(status_code=400, detail=error)
        else:
            choices = [
                {
                    "index": i,
                    "finish_reason": (
                        candidate.finish_reason.value
                        if candidate.finish_reason is not None
                        else ""
                    ),
                    "message": {
                        "content": "".join(
                            part.text
                            for part in (candidate.content.parts or [])
                            if part.text is not None
                        ),
                        "role": "assistant",
                    },
                }
                for i, candidate in enumerate(response.candidates)
                if candidate.content
            ]
            completion_response["choices"] = choices

        return completion_response


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments."""  # noqa: E501
    pg_resources = []
    pg_resources.append({"CPU": 2})  # for the deployment replica

    argparse = ArgumentParser()
    argparse.add_argument("--api_key", type=str, required=True)
    argparse.add_argument("--model_name", type=str, required=True)
    argparse.add_argument("--thinking_budget", type=int, default=1024)

    arg_strings = []
    for key, value in cli_args.items():
        if value is None:
            arg_strings.extend([f"--{key}"])
        else:
            arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)

    args = argparse.parse_args(args=arg_strings)

    logging.log(logging.INFO, f"args: {args}")

    return GeminiDeployment.options(  # type: ignore[attr-defined]
        placement_group_bundles=pg_resources,
        placement_group_strategy="STRICT_PACK",
    ).bind(
        args.api_key,
        args.model_name,
        args.thinking_budget,
    )

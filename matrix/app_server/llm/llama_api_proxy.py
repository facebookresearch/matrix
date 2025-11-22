# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from argparse import ArgumentParser
from typing import Dict, List, Optional, Union

import openai
from fastapi import FastAPI, HTTPException
from jinja2 import Template
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from ray import serve
from ray.serve import scripts
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
)

logger = logging.getLogger("ray.serve")

app = FastAPI()


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 1,
        "target_ongoing_requests": 64,
    },
    max_ongoing_requests=64,
)
@serve.ingress(app)
class LlamaApiDeployment:
    def __init__(
        self,
        api_key: str,
        model_name: str,
    ):
        self.model_name = model_name

        self.client = openai.AsyncOpenAI(
            base_url="https://api.llama.com/v1/", api_key=api_key
        )

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint."""
        logger.debug(f"Request: {request}")
        completion_request = request.model_dump(exclude_unset=True)
        completion_request.pop("guided_json", None)
        completion_request = {
            k: v for k, v in completion_request.items() if v is not None
        }

        try:
            response = await self.client.chat.completions.create(**completion_request)
            if response.choices is None and response.completion_message:
                msg = response.completion_message
                message = {
                    "role": "assistant",
                    "content": msg.get("content", {}).get("text", ""),
                }
                # Create a Choice instance
                choice = Choice(
                    index=0,
                    message=message,  # type: ignore[arg-type]
                    finish_reason=msg.get("stop_reason", ""),
                )
                response.choices = [choice]
                # Create a CompletionUsage instance
                usage_dict = {
                    m["metric"].replace("num_", ""): m["value"]
                    for m in response.metrics
                }
                usage = CompletionUsage(**usage_dict)
                response.usage = usage
            return response
        except openai.APIStatusError as e:
            detail = e.body if hasattr(e, "body") else str(e)
            raise HTTPException(status_code=e.status_code, detail=detail)
        except openai.OpenAIError as e:
            raise HTTPException(status_code=500, detail=str(e))


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments."""  # noqa: E501
    pg_resources = []
    pg_resources.append({"CPU": 2})  # for the deployment replica

    argparse = ArgumentParser()
    argparse.add_argument("--api_key", type=str, required=True)
    argparse.add_argument("--model_name", type=str, required=True)

    arg_strings = []
    for key, value in cli_args.items():
        if value is None:
            arg_strings.extend([f"--{key}"])
        else:
            arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)

    args = argparse.parse_args(args=arg_strings)

    logging.log(logging.INFO, f"args: {args}")

    return LlamaApiDeployment.options(  # type: ignore[attr-defined]
        placement_group_bundles=pg_resources,
        placement_group_strategy="STRICT_PACK",
    ).bind(
        args.api_key,
        args.model_name,
    )

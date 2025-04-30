# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import concurrent
import functools
import json
import logging
import os
from argparse import ArgumentParser
from typing import Any, Callable, Dict

import boto3
from fastapi import FastAPI, HTTPException
from ray import serve
from starlette.requests import Request
from vllm.entrypoints.openai.protocol import ChatCompletionRequest

logger = logging.getLogger("ray.serve")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

app = FastAPI()


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 1,
        "target_ongoing_requests": 64,
    },
    max_ongoing_requests=64,  # make this large so that multi-turn can route to the same replica
)
@serve.ingress(app)
class BedrockDeployment:
    def __init__(
        self,
        aws_region: str,
        model_name: str,
        anthropic_version: str,
    ):
        self.aws_region = aws_region
        self.model_name = model_name
        self.anthropic_version = anthropic_version

    def _get_client(self) -> boto3.client:
        return boto3.client(
            "bedrock-runtime",
            region_name=self.aws_region,
        )

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        completion_request = request.model_dump(exclude_unset=True)
        request_params = {
            "anthropic_version": self.anthropic_version,
            "messages": completion_request.get("messages"),
            "temperature": completion_request.get("temperature", 0.6),
            "top_p": completion_request.get("top_p", 0.9),
            "max_tokens": completion_request.get("max_tokens", 1024),
        }
        messages = completion_request.get("messages")
        if messages is not None and messages[0]["role"] == "system":
            request_params["system"] = messages[0]["content"]
            request_params["messages"].pop(0)

        client: boto3.client = self._get_client()
        loop = asyncio.get_running_loop()
        invoke_func: Callable[[], Dict[str, Any]] = functools.partial(
            client.invoke_model,
            body=json.dumps(request_params),
            modelId=self.model_name,
        )
        try:
            response: Dict[str, Any] = await loop.run_in_executor(None, invoke_func)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        completion_response = None
        if "body" in response:
            completion_response = json.loads(response["body"].read())
        if not completion_response:
            raise HTTPException(status_code=400, detail=response)
        if "content" in completion_response:
            completion_response["choices"] = [
                {
                    "finish_reason": completion_response.get("stop_reason"),
                    "index": 0,
                    "message": {
                        "content": completion_response["content"][0].get("text"),
                        "role": completion_response.get("role", "assistant"),
                    },
                }
            ]
        if "usage" in completion_response:
            usage = completion_response["usage"]
            usage["completion_tokens"] = usage.get("output_tokens", 0)
            usage["prompt_tokens"] = usage.get("input_tokens", 0)
        return completion_response


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments."""  # noqa: E501
    pg_resources = []
    pg_resources.append({"CPU": 2})  # for the deployment replica

    argparse = ArgumentParser()
    argparse.add_argument("--aws_region", type=str, default="us-west-2")
    argparse.add_argument("--model_name", type=str, required=True)
    argparse.add_argument("--anthropic_version", type=str, default="bedrock-2023-05-31")

    arg_strings = []
    for key, value in cli_args.items():
        if value is None:
            arg_strings.extend([f"--{key}"])
        else:
            arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)

    args = argparse.parse_args(args=arg_strings)

    logger.log(logging.INFO, f"args: {args}")
    assert "claude" in args.model_name.lower(), "Only Claude model is supported"

    return BedrockDeployment.options(  # type: ignore[attr-defined]
        placement_group_bundles=pg_resources,
        placement_group_strategy="STRICT_PACK",
    ).bind(
        args.aws_region,
        args.model_name,
        args.anthropic_version,
    )

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
    max_ongoing_requests=64,  # make this large so that multi-turn can route to the same replica
)
@serve.ingress(app)
class OpenaiDeployment:
    def __init__(
        self,
        api_version: str,
        api_endpoint: str,
        api_key: str,
        model_name: str,
    ):
        self.model_name = model_name
        self.is_o1 = "o1" in model_name.lower()

        if not api_endpoint.startswith("https://"):
            api_endpoint = "https://" + api_endpoint
        self.client = openai.AsyncAzureOpenAI(
            api_version=api_version, azure_endpoint=api_endpoint, api_key=api_key
        )

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
        completion_request.pop("guided_json", None)
        if self.is_o1:
            for key in ["temperature", "max_tokens", "top_p"]:
                if key in completion_request:
                    completion_request.pop(key)
        try:
            return await self.client.chat.completions.create(**completion_request)
        except openai.APIStatusError as e:
            detail = e.body if hasattr(e, "body") else str(e)
            raise HTTPException(status_code=e.status_code, detail=detail)
        except openai.OpenAIError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/completions")
    async def create_completion(self, request: CompletionRequest, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        logger.debug(f"Request: {request}")
        completion_request = request.model_dump(exclude_unset=True)
        completion_request.pop("guided_json", None)
        try:
            return await self.client.completions.create(**completion_request)
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
    argparse.add_argument("--api_version", type=str, required=True)
    argparse.add_argument("--api_key", type=str, required=True)
    argparse.add_argument("--api_endpoint", type=str, required=True)
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

    return OpenaiDeployment.options(  # type: ignore[attr-defined]
        placement_group_bundles=pg_resources,
        placement_group_strategy="STRICT_PACK",
    ).bind(
        args.api_version,
        args.api_endpoint,
        args.api_key,
        args.model_name,
    )

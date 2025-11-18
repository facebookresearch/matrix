# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from argparse import ArgumentParser
from typing import Dict, List, Optional, Union

import aiohttp
from fastapi import FastAPI, HTTPException
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

from matrix.utils.http import post_url
from matrix.utils.logging import get_logger

logger = get_logger("ray.serve")

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
class MetagenDeployment:
    def __init__(
        self,
        access_token: str,
        model_name: str,
    ):
        self.model_name = model_name
        self.url_base = "https://graph.facebook.com/v22.0"
        self.endpoint = "chat_completions"
        self.access_token = access_token

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint."""
        logger.debug(f"Request: {request}")
        completion_request = request.model_dump(exclude_unset=True)
        messages = completion_request.pop("messages")
        messages = [{"role": msg["role"], "text": msg["content"]} for msg in messages]
        model = self.model_name
        request_params = {
            "access_token": self.access_token,
            "model": model,
            "messages": messages,
            "options": {  # For available options see https://fburl.com/code/6u69j4hm
                "temperature": completion_request.get(
                    "temperature", 0.6
                ),  # default is 0.6
                "top_p": completion_request.get("top_p", 0.9),  # default is 0.9
                "decode_output": True,  # default is True,
                # "custom_stop": {"stop_words": [{"text": "ch", "token": 331}]},  # default is None
            },
        }

        async with aiohttp.ClientSession() as session:
            status, content = await post_url(session, f"{self.url_base}/{self.endpoint}", request_params)  # type: ignore
        if status is None:
            raise Exception(f"Error querying metagen model: {content}")
        response = json.loads(content)
        if (error := response.get("error")) is not None:
            if error.get("is_transient"):
                headers = {"Retry-After": str(60)}
                raise HTTPException(status_code=429, detail=error, headers=headers)
            else:
                raise HTTPException(status_code=status, detail=error)

        text = response.get("text")
        if not text:
            raise HTTPException(status_code=400, detail=response)
        usage = response.get("usage", {})
        completion_response = {
            "id": response.get("response_id"),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": response.get("finish_reason"),
                    "message": {"content": text, "role": "assistant"},
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("num_prompt_tokens"),
                "total_tokens": usage.get("num_total_tokens"),
                "completion_tokens": usage.get("num_completion_tokens"),
            },
        }

        return completion_response


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments."""  # noqa: E501
    pg_resources = []
    pg_resources.append({"CPU": 2})  # for the deployment replica

    argparse = ArgumentParser()
    argparse.add_argument("--access_token", type=str, required=True)
    argparse.add_argument("--model_name", type=str, required=True)

    arg_strings = []
    for key, value in cli_args.items():
        if value is None:
            arg_strings.extend([f"--{key}"])
        else:
            arg_strings.extend([f"--{key}", str(value)])
    logger.info(",".join(arg_strings))

    args = argparse.parse_args(args=arg_strings)

    logging.log(logging.INFO, f"args: {args}")

    return MetagenDeployment.options(  # type: ignore[attr-defined]
        placement_group_bundles=pg_resources,
        placement_group_strategy="STRICT_PACK",
    ).bind(
        args.access_token,
        args.model_name,
    )

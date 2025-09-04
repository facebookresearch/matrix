# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import concurrent
import functools
import json
import logging
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
class SageMakerDeployment:
    def __init__(
        self,
        aws_account: str,
        aws_region: str,
        endpoint_name: str,
        model_name: str,
    ):
        self.aws_account = aws_account
        self.aws_region = aws_region
        self.endpoint_name = endpoint_name
        self.model_name = model_name

        sts = boto3.client("sts")
        role = sts.assume_role(
            RoleArn=f"arn:aws:iam::{self.aws_account}:role/SageMakerExternal",
            RoleSessionName="matrix_session",
        )
        self.creds: Dict[str, Any] = role["Credentials"]

    def _get_client(self) -> boto3.client:
        return boto3.client(
            "sagemaker-runtime",
            region_name=self.aws_region,
            aws_access_key_id=self.creds["AccessKeyId"],
            aws_secret_access_key=self.creds["SecretAccessKey"],
            aws_session_token=self.creds["SessionToken"],
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
            "model": self.model_name,
            "messages": completion_request.get("messages"),
            "temperature": completion_request.get("temperature", 0.6),
            "top_p": completion_request.get("top_p", 0.9),
            "max_tokens": completion_request.get("max_tokens", 1024),
            "n": completion_request.get("n", 1),
            "logprobs": completion_request.get("logprobs", False),
        }

        client: boto3.client = self._get_client()
        loop = asyncio.get_running_loop()
        invoke_func: Callable[[], Dict[str, Any]] = functools.partial(
            client.invoke_endpoint,
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=bytes(json.dumps(request_params), "utf-8"),
        )
        response: Dict[str, Any] = await loop.run_in_executor(None, invoke_func)

        metadata: Dict[str, str | int] = response["ResponseMetadata"]
        if (metadata.get("HTTPStatusCode")) != 200:
            raise HTTPException(status_code=400, detail=response)

        completion_response: Dict[str, str] = json.loads(
            response["Body"].read().decode("utf-8")
        )
        if not completion_response:
            raise HTTPException(status_code=400, detail=response)
        return completion_response


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments."""  # noqa: E501
    pg_resources = []
    pg_resources.append({"CPU": 2})  # for the deployment replica

    argparse = ArgumentParser()
    argparse.add_argument("--aws_account", type=str, required=True)
    argparse.add_argument("--aws_region", type=str, required=True)
    argparse.add_argument("--endpoint_name", type=str, required=True)
    argparse.add_argument("--model_name", type=str, required=True)

    arg_strings = []
    for key, value in cli_args.items():
        if value is None:
            arg_strings.extend([f"--{key}"])
        else:
            arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)

    args = argparse.parse_args(args=arg_strings)

    logger.log(logging.INFO, f"args: {args}")

    return SageMakerDeployment.options(  # type: ignore[attr-defined]
        placement_group_bundles=pg_resources,
        placement_group_strategy="STRICT_PACK",
    ).bind(
        args.aws_account,
        args.aws_region,
        args.endpoint_name,
        args.model_name,
    )

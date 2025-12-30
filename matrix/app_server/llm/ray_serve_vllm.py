# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from asyncio import Lock
from inspect import Parameter, signature
from typing import Any, Dict, List, Optional, Union

import fire
import grpc
import vllm
import yaml
from fastapi import FastAPI
from google.protobuf import json_format
from jinja2 import Template
from ray import serve
from ray.serve import scripts
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs

try:
    from vllm.engine.async_llm_engine import AsyncEngineDeadError, AsyncLLMEngine

    _has_v0 = True
except ImportError:
    _has_v0 = False
    from vllm.v1.engine.async_llm import EngineDeadError as AsyncEngineDeadError

try:
    from vllm.v1.engine.async_llm import AsyncLLM

    _has_v1 = True
except ImportError:
    _has_v1 = False
try:
    from vllm.engine.metrics import RayPrometheusStatLogger
except ImportError:
    from vllm.v1.metrics.ray_wrappers import RayPrometheusStatLogger

from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
)

from matrix.app_server.llm import openai_pb2

try:
    from vllm.entrypoints.openai.serving_engine import (  # type: ignore[attr-defined]
        BaseModelPath,
    )

    has_base_model_path = True
except:
    try:
        from vllm.entrypoints.openai.serving_models import (  # type: ignore[no-redef]
            BaseModelPath,
        )

        has_base_model_path = True
    except:
        has_base_model_path = False
from vllm.config import ModelConfig
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion

try:
    from vllm.entrypoints.openai.serving_engine import (  # type: ignore[attr-defined]
        LoRAModulePath,
    )
except:
    from vllm.entrypoints.openai.serving_models import (
        LoRAModulePath,  # type: ignore[no-redef]
    )

try:
    from vllm.utils import FlexibleArgumentParser
except:
    from vllm.utils.argparse_utils import FlexibleArgumentParser

vllm_deploy_args = [
    "use_v1_engine",
    "enable_tools",
]

logger = logging.getLogger("ray.serve")

app = FastAPI()


def use_ray_executor(cls, engine_config):
    logger.info("Force ray executor")
    try:
        from vllm.executor.ray_gpu_executor import RayGPUExecutorAsync

        return RayGPUExecutorAsync
    except:
        from vllm.executor.ray_distributed_executor import RayDistributedExecutor

        return RayDistributedExecutor


from vllm.config import DeviceConfig

# Save original method
original_post_init = getattr(DeviceConfig, "__post_init__", None)
if original_post_init is not None:

    def patched_post_init(self):
        try:
            original_post_init(self)  # type: ignore[misc]
        except Exception as e:
            print(f"[Patch] Device detection failed: {e}, defaulting to 'cuda'")
            import torch

            self.device_type = "cuda"
            self.device = torch.device("cuda")

    DeviceConfig.__post_init__ = patched_post_init  # type: ignore[attr-defined]


class BaseDeployment:
    lora_modules: Optional[List[LoRAModulePath]] = None
    use_v1_engine: Optional[bool] = None
    enable_tools: bool = False
    tool_parser: Optional[str] = None

    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        request_logger: Optional[RequestLogger] = None,
        chat_template: Optional[str] = None,
        tool_parser: Optional[str] = None,
        enable_tools: bool = False,
        use_v1_engine: Optional[bool] = None,
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        # self.openai_serving_chat = None
        # self.openai_serving_completion = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.request_logger = request_logger
        self.chat_template = chat_template
        self.use_v1_engine = (
            _has_v1 and use_v1_engine is not None and use_v1_engine == True
        ) or not _has_v0
        self.enable_tools = enable_tools
        self.tool_parser = tool_parser
        # AsyncLLMEngine._get_executor_cls = classmethod(use_ray_executor)

        # current_platform.get_device_capability() would return None for some models (e.g. R1) after
        # bump vllm to 0.7.3. This line is the fix, according to https://github.com/vllm-project/vllm/issues/8402#issuecomment-2489432973
        # related issues
        # https://github.com/vllm-project/vllm/issues/8402
        # https://github.com/vllm-project/vllm/issues/7890
        # https://github.com/ray-project/ray/pull/51007
        del os.environ["CUDA_VISIBLE_DEVICES"]

        # increase the timeout of getting result from a compiled graph execution
        # https://github.com/vllm-project/vllm/pull/15301
        if engine_args.pipeline_parallel_size > 1:
            os.environ["RAY_CGRAPH_get_timeout"] = "1200"

        if self.use_v1_engine:
            os.environ["VLLM_USE_V1"] = "1"
            os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
            # might rm ~/.cache/vllm if this causing issues
            # os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

        if self.use_v1_engine:
            self.engine = AsyncLLM.from_engine_args(engine_args)
        else:
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)  # type: ignore[assignment]
        self.create_openai()
        if hasattr(self.engine, "add_logger"):
            #  only AsyncLLMEngine
            self.create_prometheus_logger()

    @serve.multiplexed(max_num_models_per_replica=500)
    async def get_model(self, model_id: str):
        return model_id

    def create_openai(
        self,
    ):
        if hasattr(self.engine, "model_config"):
            model_config = self.engine.model_config
        else:
            model_config = self.engine.engine.get_model_config()  # type: ignore[attr-defined]

        init_params = signature(OpenAIServingChat.__init__).parameters

        # Prepare arguments dynamically based on detected parameters
        kwargs = {
            "engine_client": self.engine,
            "request_logger": self.request_logger,
            "chat_template": self.chat_template,
            "response_role": self.response_role,
        }
        if self.engine_args.served_model_name is not None:
            base_model_paths = [
                BaseModelPath(name=name, model_path=self.engine_args.model)
                for name in self.engine_args.served_model_name
            ]
        else:
            if has_base_model_path:
                base_model_paths = [
                    BaseModelPath(self.engine_args.model, self.engine_args.model)  # type: ignore[list-item]
                ]
            else:
                base_model_paths = [self.engine_args.model]

        # v0.7.0
        if "models" in init_params:
            from vllm.entrypoints.openai.serving_models import OpenAIServingModels

            model_kwargs = {
                "engine_client": self.engine,
                "base_model_paths": base_model_paths,
                "lora_modules": self.lora_modules,
            }
            if "model_config" in signature(OpenAIServingModels.__init__).parameters:
                model_kwargs["model_config"] = model_config
            # New version: Use `models` and `chat_template_content_format`
            kwargs["models"] = OpenAIServingModels(**model_kwargs)
        if "chat_template_content_format" in init_params:
            kwargs["chat_template_content_format"] = "auto"

        # v0.6.6
        if "lora_modules" in init_params:
            kwargs["lora_modules"] = self.lora_modules
        if "base_model_paths" in init_params:
            kwargs["base_model_paths"] = base_model_paths

        # equivalent to --enable-auto-tool-choice
        if self.enable_tools:
            kwargs["enable_auto_tools"] = True
            kwargs["tool_parser"] = self.tool_parser

        if "model_config" in init_params:
            kwargs["model_config"] = model_config

        self.openai_serving_chat = OpenAIServingChat(**kwargs)  # type: ignore[arg-type]
        completion_exclude = [
            "chat_template",
            "chat_template_content_format",
            "response_role",
            "enable_auto_tools",
            "tool_parser",
        ]
        self.openai_serving_completion = OpenAIServingCompletion(
            **{k: v for k, v in kwargs.items() if not k in completion_exclude}  # type: ignore[arg-type]
        )

    def create_prometheus_logger(
        self,
    ):
        init_params = signature(RayPrometheusStatLogger.__init__).parameters
        kwargs = {
            "local_interval": 5,
            "labels": dict(model_name=self.engine_args.model),
        }
        # v0.7.0
        if "vllm_config" in init_params:
            kwargs["vllm_config"] = self.engine.engine.vllm_config  # type: ignore[attr-defined]

        # v0.6.6
        if "max_model_len" in init_params:
            model_config = self.engine.engine.get_model_config()  # type: ignore[attr-defined]
            kwargs["max_model_len"] = model_config.max_model_len

        additional_metrics_logger: RayPrometheusStatLogger = RayPrometheusStatLogger(
            **kwargs  # type: ignore[arg-type]
        )
        self.engine.add_logger("ray", additional_metrics_logger)  # type: ignore[attr-defined]


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 8,
        "target_ongoing_requests": 64,
    },
    max_ongoing_requests=64,  # make this large so that multi-turn can route to the same replica
)
@serve.ingress(app)
class VLLMDeployment(BaseDeployment):
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        request_logger: Optional[RequestLogger] = None,
        chat_template: Optional[str] = None,
        tool_parser: Optional[str] = None,
        enable_tools: bool = False,
        use_v1_engine: Optional[bool] = None,
    ):
        super().__init__(
            engine_args=engine_args,
            response_role=response_role,
            lora_modules=lora_modules,
            request_logger=request_logger,
            chat_template=chat_template,
            tool_parser=tool_parser,
            enable_tools=enable_tools,
            use_v1_engine=use_v1_engine,
        )

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        model_id = serve.get_multiplexed_model_id()
        if model_id:
            model = await self.get_model(model_id)
        logger.debug(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            if hasattr(generator, "error"):
                generator = generator.error
            return JSONResponse(
                content=generator.model_dump(exclude_unset=True, exclude_none=True),
                status_code=generator.code,
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")  # type: ignore[arg-type]
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(
                content=generator.model_dump(exclude_unset=True, exclude_none=True)
            )

    @app.post("/v1/completions")
    async def create_completion(self, request: CompletionRequest, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        logger.debug(f"Request: {request}")
        generator = await self.openai_serving_completion.create_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            if hasattr(generator, "error"):
                generator = generator.error
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")  # type: ignore[arg-type]
        else:
            assert isinstance(generator, CompletionResponse)
            return JSONResponse(
                content=generator.model_dump(exclude_unset=True, exclude_none=True)
            )


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 8,
        "target_ongoing_requests": 64,
    },
    max_ongoing_requests=64,  # make this large so that multi-turn can route to the same replica
)
class GrpcDeployment(BaseDeployment):
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        request_logger: Optional[RequestLogger] = None,
        chat_template: Optional[str] = None,
        tool_parser: Optional[str] = None,
        enable_tools: bool = False,
        use_v1_engine: Optional[bool] = None,
    ):
        super().__init__(
            engine_args=engine_args,
            response_role=response_role,
            lora_modules=lora_modules,
            request_logger=request_logger,
            chat_template=chat_template,
            tool_parser=tool_parser,
            enable_tools=enable_tools,
            use_v1_engine=use_v1_engine,
        )
        self.healthy = True

    async def check_health(self):
        if self.healthy:
            return {"status": "healthy"}
        else:
            raise RuntimeError("Replica unhealthy!")  # Triggers Ray Serve restart

    def http_to_grpc_status(self, http_status_code: int) -> grpc.StatusCode:
        """A simple function to map HTTP status codes to gRPC status codes."""
        mapping = {
            400: grpc.StatusCode.INVALID_ARGUMENT,
            401: grpc.StatusCode.UNAUTHENTICATED,
            403: grpc.StatusCode.PERMISSION_DENIED,
            404: grpc.StatusCode.NOT_FOUND,
            409: grpc.StatusCode.ALREADY_EXISTS,
            429: grpc.StatusCode.RESOURCE_EXHAUSTED,
            499: grpc.StatusCode.CANCELLED,
            500: grpc.StatusCode.INTERNAL,
            501: grpc.StatusCode.UNIMPLEMENTED,
            502: grpc.StatusCode.UNAVAILABLE,
            503: grpc.StatusCode.UNAVAILABLE,
            504: grpc.StatusCode.DEADLINE_EXCEEDED,
        }
        return mapping.get(http_status_code, grpc.StatusCode.UNKNOWN)

    async def CreateChatCompletion(self, request):
        """OpenAI-compatible GRPC endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        model_id = serve.get_multiplexed_model_id()
        if model_id:
            model = await self.get_model(model_id)

        chat = ChatCompletionRequest(
            **json_format.MessageToDict(request, preserving_proto_field_name=True)
        )
        logger.debug(f"Request: {chat}")
        try:
            if (
                hasattr(self.openai_serving_chat, "models")
                and self.openai_serving_chat.models.static_lora_modules
                and len(self.openai_serving_chat.models.lora_requests) == 0
            ):
                # only need for lora modules, at vllm >= v0.7.0
                # due to https://github.com/vllm-project/vllm/commit/ac2f3f7fee93cf9cd97c0078e362feab7b6c8299
                await self.openai_serving_chat.models.init_static_loras()
            generator = await self.openai_serving_chat.create_chat_completion(chat)
            if isinstance(generator, ErrorResponse):
                if hasattr(generator, "error"):
                    generator = generator.error
                status_code = self.http_to_grpc_status(generator.code)
                raise grpc.RpcError(
                    status_code,
                    generator.model_dump(exclude_unset=True, exclude_none=True),
                )

            assert isinstance(generator, ChatCompletionResponse)
            response = openai_pb2.ChatCompletionResponse()  # type: ignore[attr-defined]
            response_dict = generator.model_dump(
                exclude_unset=True,
                exclude_none=True,
            )
            for choice in response_dict["choices"]:
                if "stop_reason" in choice:
                    choice["stop_reason"] = str(choice["stop_reason"])
                if "reasoning_content" in choice.get("message", {}):
                    choice["message"].pop(
                        "reasoning", None
                    )  # gpt-oss has reasoning, duplicate of reasoning_content
            json_format.ParseDict(response_dict, response)
            return response
        except AsyncEngineDeadError as e:
            self.healthy = False
            logger.info(f"vLLM Engine Dead: {e}")
            raise RuntimeError("vLLM Engine has dead and needs restarting.") from e

    async def CreateCompletion(self, request):
        """OpenAI-compatible GRPC endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        model_id = serve.get_multiplexed_model_id()
        if model_id:
            model = await self.get_model(model_id)
        completion_request = CompletionRequest(
            **json_format.MessageToDict(request, preserving_proto_field_name=True)
        )
        logger.debug(f"Request: {completion_request}")
        try:
            if (
                self.openai_serving_chat.models.static_lora_modules
                and len(self.openai_serving_chat.models.lora_requests) == 0
            ):
                # only need for lora modules, at vllm >= v0.7.0
                # due to https://github.com/vllm-project/vllm/commit/ac2f3f7fee93cf9cd97c0078e362feab7b6c8299
                await self.openai_serving_chat.models.init_static_loras()
            generator = await self.openai_serving_completion.create_completion(
                completion_request,
            )
            if isinstance(generator, ErrorResponse):
                if hasattr(generator, "error"):
                    generator = generator.error
                status_code = self.http_to_grpc_status(generator.code)
                raise grpc.RpcError(
                    status_code,
                    generator.model_dump(exclude_unset=True, exclude_none=True),
                )

            assert isinstance(generator, CompletionResponse)
            response = openai_pb2.CompletionResponse()  # type: ignore[attr-defined]
            response_dict = generator.model_dump(
                exclude={"top_logprobs"},  # type: ignore[arg-type]
                exclude_unset=True,
                exclude_none=True,
            )
            for choice in response_dict["choices"]:
                if "stop_reason" in choice:
                    choice["stop_reason"] = str(choice["stop_reason"])
                if "logprobs" in choice and "top_logprobs" in choice["logprobs"]:
                    choice["logprobs"].pop("top_logprobs")
                if "prompt_logprobs" in choice:
                    for index, logprobs in enumerate(choice["prompt_logprobs"]):
                        choice["prompt_logprobs"][index] = {"token_map": logprobs or {}}
            json_format.ParseDict(response_dict, response)
            return response
        except AsyncEngineDeadError as e:
            self.healthy = False
            logger.info(f"vLLM Engine Dead: {e}")
            raise RuntimeError("vLLM Engine has dead and needs restarting.") from e


def parse_vllm_args(cli_args: Dict[str, str]):
    """Parses vLLM args based on CLI inputs.

    Currently uses argparse because vLLM doesn't expose Python models for all of the
    config options we want to support.
    """
    arg_parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )

    parser = make_arg_parser(arg_parser)
    arg_strings = []
    deploy_args = {}
    for key, value in cli_args.items():
        if key in vllm_deploy_args:
            deploy_args[key] = value
        else:
            if value is None:
                arg_strings.extend([f"--{key}"])
            else:
                arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args, deploy_args


def _build_app(cli_args: Dict[str, Any], use_grpc) -> serve.Application:
    """Builds the Serve app based on CLI arguments.

    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server
    for the complete set of arguments.

    Supported engine arguments: https://docs.vllm.ai/en/latest/models/engine_args.html.
    """  # noqa: E501
    ray_resources: Dict[str, Any] = cli_args.pop("ray_resources", {})
    accelerator = "GPU"
    cli_args["distributed-executor-backend"] = "ray"
    parsed_args, deploy_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)

    tp = engine_args.tensor_parallel_size
    pp = engine_args.pipeline_parallel_size
    logger.info(f"Tensor parallelism = {tp}, Pipeline parallelism = {pp}")
    pg_resources = []
    pg_resources.append({"CPU": 1})  # for the deployment replica
    if ray_resources.get("num_gpus", 1) != 1:
        logger.warning("GPU must be 1")
    for i in range(tp * pp):
        pg_resources.append(
            {"CPU": ray_resources.get("num_cpus", 4), accelerator: 1}
        )  # for the vLLM actors

    # We use the "STRICT_PACK" strategy below to ensure all vLLM actors are placed on
    # the same Ray node.
    cls = VLLMDeployment if not use_grpc else GrpcDeployment
    return cls.options(  # type: ignore[union-attr]
        placement_group_bundles=pg_resources,
        placement_group_strategy="STRICT_PACK" if pp == 1 else "PACK",
    ).bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        cli_args.get("request_logger"),
        parsed_args.chat_template,
        parsed_args.tool_call_parser,
        **deploy_args,
    )


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    return _build_app(cli_args, use_grpc=False)


def build_app_grpc(cli_args: Dict[str, str]) -> serve.Application:
    return _build_app(cli_args, use_grpc=True)

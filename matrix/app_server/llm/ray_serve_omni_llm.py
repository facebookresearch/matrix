# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("ray.serve")

app = FastAPI()

# Args specific to omni deployment (not vllm engine args)
omni_deploy_args = ["deploy_config", "async_chunk", "stage_overrides", "num_gpus"]


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 8,
        "target_ongoing_requests": 64,
    },
    max_ongoing_requests=64,
)
@serve.ingress(app)
class OmniDeployment:
    def __init__(
        self,
        model: str,
        deploy_config: Optional[str] = None,
        async_chunk: bool = True,
        stage_overrides: Optional[Dict[str, Any]] = None,
        num_gpus: int = 2,
        served_model_name: Optional[List[str]] = None,
        response_role: str = "assistant",
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        # Lazy imports for vllm-omni dependencies
        from vllm_omni import AsyncOmni

        logger.info(f"Starting OmniDeployment with model: {model}")

        # NOTE: Unlike standard vLLM deployments, we do NOT delete CUDA_VISIBLE_DEVICES.
        # vllm-omni uses multiproc_executor (local subprocesses, not Ray actors) and
        # relies on CUDA_VISIBLE_DEVICES for logical→physical device mapping across
        # its multi-stage pipeline (thinker/talker/code2wav).

        self.model = model
        self.response_role = response_role

        # Build AsyncOmni engine
        # NOTE: We do NOT pass tensor_parallel_size / pipeline_parallel_size to
        # AsyncOmni. vllm-omni's stage YAML controls per-stage TP/PP and device
        # mapping. num_gpus is only used for placement group GPU allocation.
        omni_kwargs: Dict[str, Any] = {"model": model}
        if deploy_config:
            # AsyncOmni expects 'stage_configs_path', not 'deploy_config'
            omni_kwargs["stage_configs_path"] = deploy_config
        if stage_overrides and "stage_configs_path" not in omni_kwargs:
            # Apply per-stage engine_args overrides by generating a patched
            # stage config YAML.  AsyncOmni doesn't support stage_overrides
            # natively, so we resolve the default YAML, patch it, and pass
            # the patched file via stage_configs_path.
            patched = self._apply_stage_overrides(model, stage_overrides)
            if patched:
                omni_kwargs["stage_configs_path"] = patched
        omni_kwargs["async_chunk"] = async_chunk
        # Pass through any extra engine args
        for k, v in kwargs.items():
            omni_kwargs[k] = v

        self.engine = AsyncOmni(**omni_kwargs)

        # Set up model names for OpenAI-compatible serving
        self.served_model_name = served_model_name or [model]

        # Set up OpenAI-compatible chat serving
        self._setup_chat_serving(chat_template)

        # Set up speech serving from vllm_omni
        self._setup_speech_serving()

    def _setup_chat_serving(self, chat_template: Optional[str]):
        from vllm.entrypoints.openai.models.protocol import BaseModelPath
        from vllm.entrypoints.openai.models.serving import OpenAIServingModels
        from vllm.entrypoints.serve.render.serving import OpenAIServingRender
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

        base_model_paths = [
            BaseModelPath(name=name, model_path=self.model)
            for name in self.served_model_name
        ]

        models = OpenAIServingModels(
            engine_client=self.engine,
            base_model_paths=base_model_paths,
        )

        render = OpenAIServingRender(
            model_config=self.engine.model_config,
            renderer=self.engine.renderer,
            io_processor=self.engine.io_processor,
            model_registry=models.registry,
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format="auto",
        )

        self.openai_serving_chat = OmniOpenAIServingChat(
            self.engine,
            models,
            self.response_role,
            openai_serving_render=render,
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format="auto",
        )

    def _setup_speech_serving(self):
        try:
            from vllm.entrypoints.openai.models.protocol import BaseModelPath
            from vllm.entrypoints.openai.models.serving import OpenAIServingModels
            from vllm_omni.entrypoints.openai.serving_speech import (
                OmniOpenAIServingSpeech,
            )

            base_model_paths = [
                BaseModelPath(name=name, model_path=self.model)
                for name in self.served_model_name
            ]
            models = OpenAIServingModels(
                engine_client=self.engine,
                base_model_paths=base_model_paths,
            )
            self.openai_serving_speech = OmniOpenAIServingSpeech(
                self.engine, models, request_logger=None
            )
            self._has_speech = True
            logger.info("Speech serving enabled")
        except Exception:
            self.openai_serving_speech = None
            self._has_speech = False
            logger.warning(
                "vllm_omni speech serving not available, /v1/audio/speech disabled",
                exc_info=True,
            )

    @staticmethod
    def _apply_stage_overrides(
        model: str, stage_overrides: Dict[str, Any]
    ) -> Optional[str]:
        """Generate a patched stage config YAML with per-stage overrides applied.

        Args:
            model: Model name or path for resolving the default stage config.
            stage_overrides: Mapping of stage_id (int) -> dict of engine_args
                to override.  Example: {1: {"enforce_eager": True}}

        Returns:
            Path to the patched temporary YAML file, or None on failure.
        """
        from vllm_omni.entrypoints.utils import resolve_model_config_path

        config_path = resolve_model_config_path(model)
        if config_path is None:
            logger.warning(
                "Could not resolve stage config for model %s; "
                "stage_overrides will be ignored",
                model,
            )
            return None

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        stage_key = "stage_args" if "stage_args" in config else "stages"
        stages = config.get(stage_key, [])

        for stage in stages:
            sid = stage.get("stage_id")
            overrides = stage_overrides.get(sid)
            if overrides is None:
                continue
            engine_args = stage.setdefault("engine_args", {})
            engine_args.update(overrides)
            logger.info(
                "stage_overrides: stage %s engine_args patched with %s",
                sid,
                overrides,
            )

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="omni_stage_", delete=False
        )
        yaml.dump(config, tmp, default_flow_style=False)
        tmp.flush()
        tmp.close()
        logger.info("Patched stage config written to %s", tmp.name)
        return tmp.name

    async def check_health(self):
        await self.engine.check_health()
        capabilities = ["chat"]
        if self._has_speech:
            capabilities.extend(["tts", "asr", "speech_to_speech"])
        return {"status": "healthy", "capabilities": capabilities}

    @app.post("/v1/chat/completions")
    async def create_chat_completion(self, request: Request):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
            ChatCompletionResponse,
        )
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse

        body = await request.json()
        chat_request = ChatCompletionRequest(**body)

        logger.debug(f"Chat request: {chat_request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            chat_request, request
        )
        if isinstance(generator, ErrorResponse):
            if hasattr(generator, "error"):
                generator = generator.error
            return JSONResponse(
                content=generator.model_dump(exclude_unset=True, exclude_none=True),
                status_code=generator.code,
            )
        if chat_request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(
                content=generator.model_dump(exclude_unset=True, exclude_none=True)
            )

    @app.post("/v1/audio/speech")
    async def create_speech(self, raw_request: Request):
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse
        from vllm_omni.entrypoints.openai.protocol.audio import (
            OpenAICreateSpeechRequest,
        )

        if not self._has_speech:
            return JSONResponse(
                content={"error": "Speech endpoint not available"},
                status_code=501,
            )

        body = await raw_request.json()
        request = OpenAICreateSpeechRequest(**body)
        result = await self.openai_serving_speech.create_speech(request, raw_request)
        if isinstance(result, ErrorResponse):
            return JSONResponse(
                content=result.model_dump(),
                status_code=result.code if hasattr(result, "code") else 400,
            )
        return result


def parse_omni_args(cli_args: Dict[str, Any]):
    """Parse CLI args for omni deployment, separating deploy-specific args from engine args."""
    deploy_args: Dict[str, Any] = {}
    engine_args: Dict[str, Any] = {}
    for key, value in cli_args.items():
        if key in omni_deploy_args:
            deploy_args[key] = value
        else:
            engine_args[key.replace("-", "_")] = value
    return engine_args, deploy_args


def build_omni_app(cli_args: Dict[str, Any]) -> serve.Application:
    """Builds the Serve app for omni (speech) models.

    Supports the same CLI arguments as vLLM plus omni-specific args
    like deploy_config, async_chunk, and stage_overrides.

    Unlike standard vLLM, vllm-omni uses multiproc_executor (local subprocesses)
    rather than Ray distributed executor. All GPUs must be directly visible to the
    deployment replica, so we allocate them in a single placement group bundle.
    """
    ray_resources: Dict[str, Any] = cli_args.pop("ray_resources", {})
    accelerator = "GPU"

    engine_args, deploy_args = parse_omni_args(cli_args)

    num_gpus = int(deploy_args.pop("num_gpus", 2))
    logger.info(f"Omni: Total GPUs = {num_gpus}")

    # vllm-omni manages multi-stage GPU allocation internally via multiproc_executor.
    # Unlike standard vLLM (which uses Ray actors for TP workers), we must give the
    # deployment replica direct access to all GPUs in a single bundle.
    pg_resources = [
        {"CPU": ray_resources.get("num_cpus", 1), accelerator: num_gpus},
    ]

    ray_resources.pop("num_cpus", None)
    ray_resources.pop("num_gpus", None)
    custom_resources = ray_resources.pop("resources", None)

    if custom_resources:
        for bundle in pg_resources:
            bundle.update(custom_resources)

    # Merge deploy_args into engine_args for the constructor
    all_args = {**engine_args, **deploy_args}

    return OmniDeployment.options(
        placement_group_bundles=pg_resources,
        placement_group_strategy="STRICT_PACK",
        # The actor must request GPUs so Ray sets CUDA_VISIBLE_DEVICES to the
        # allocated devices. Without this, CUDA_VISIBLE_DEVICES is empty.
        ray_actor_options={"num_gpus": num_gpus},
        **ray_resources,
    ).bind(**all_args)

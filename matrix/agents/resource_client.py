# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import os
import random
import traceback
from typing import Any, Dict, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

from matrix.client import query_llm
from matrix.client.container_client import ContainerClient

from .orchestrator import BaseResourceClient

logger = logging.getLogger(__name__)


class LLMResourceClient(BaseResourceClient):
    def __init__(
        self,
        resource_id: str,
        matrix_cli,
        matrix_service: str,
        sampling_params: Optional[Dict[str, Any]] = None,
        tools_params: Optional[Dict[str, Any]] = None,
        exec_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(resource_id)
        self.sampling_params = sampling_params or {}
        self.tools_params = tools_params or {}
        self.exec_params = exec_params or {}
        self.llm_metadata = matrix_cli.get_app_metadata(matrix_service)

        self.endpoint_cache = (
            self.llm_metadata["endpoints"]["updater"]
            if self.llm_metadata and "endpoints" in self.llm_metadata
            else None
        )

    async def utilize(self, resource_info, logger, messages: dict[list, Any], seed, task_id, **kwargs):  # type: ignore[override]
        logger.debug(f"Calling with {self.resource_id} {messages}")
        exec_params = self.exec_params.copy()
        multiplexed_model_id = exec_params.pop("sticky_routing_prefix", None)
        if multiplexed_model_id:
            exec_params["multiplexed_model_id"] = f"{multiplexed_model_id}{task_id}"
        try:
            result = await query_llm.make_request(
                url=None,
                model=self.llm_metadata["model_name"],
                app_name=self.llm_metadata["name"],
                data={"messages": messages},
                endpoint_cache=self.endpoint_cache,
                **self.sampling_params,
                **self.tools_params,
                **exec_params,
                seed=seed,
                **kwargs,
            )
            response = result["response"]
        except Exception as e:
            tb_str = traceback.format_exc()
            msg = f"Task failed {repr(e)} {tb_str}"
            logger.error(msg)
            response = {"error": msg}
        return response


class ContainerResourceClient(BaseResourceClient):
    def __init__(
        self,
        resource_id,
        matrix_service: str,
        matrix_cli,
        start_config: Optional[Dict[str, Any]] = None,
        exec_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(resource_id)
        self.start_config: Dict[str, Any] = start_config or {}
        self.exec_params: Dict[str, Any] = exec_params or {}
        if isinstance(start_config, (DictConfig, ListConfig)):
            self.start_config = OmegaConf.to_container(start_config, resolve=True)
        if isinstance(exec_params, (DictConfig, ListConfig)):
            self.exec_params = OmegaConf.to_container(exec_params, resolve=True)
        app_metadata = matrix_cli.get_app_metadata(matrix_service)
        base_url = app_metadata["endpoints"]["head"]
        timeout_secs = self.exec_params.get("timeout_secs")
        self.container_client = ContainerClient(base_url, timeout=timeout_secs)

    async def acquire(self, task: Dict[str, Any], logger):
        # allocate the container, crash if containers are not available
        max_retries = self.exec_params.get(
            "acquire_max_retries", 1 << 31
        )  # wait for when container is available
        initial_delay = 1
        backoff_factor = 2

        bind_dir = self.start_config.get("bind_dir")
        if bind_dir is not None:
            bind_dir = os.path.abspath(os.path.expanduser(bind_dir))

        logger.debug(f"Acquire container: {self.start_config}")
        container_config = {
            "image": (
                self.get_container_image(task)
                if "image" not in self.start_config
                else os.path.abspath(os.path.expanduser(self.start_config["image"]))
            ),
            "run_args": self.start_config.get("run_args", [])
            + (["--bind", f"{bind_dir}:{bind_dir}"] if bind_dir is not None else []),
            "start_script_args": self.start_config.get("start_script_args"),
        }

        logger.debug(f"Acquiring container for {container_config}")
        for attempt in range(max_retries):
            container_info = await self.container_client.acquire_container(
                **container_config
            )
            if "error" in container_info:
                if "retry" in container_info["error"]:
                    if attempt < max_retries - 1:
                        delay = initial_delay * (
                            backoff_factor**attempt + random.uniform(0, 1)
                        )
                        logger.debug(f"Waiting to acquire container attempt {attempt}")
                        await asyncio.sleep(delay)
                        continue
                raise Exception(
                    f"Failed to acquire container: {container_info['error']}"
                )
            else:
                break
        logger.debug(f"Acquired container {container_info} for {container_config}")
        return container_info

    async def release(self, resource_info: dict[str, Any], logger):
        container_id = resource_info["container_id"]
        logger.debug(f"Releasing container {container_id}")
        result = await self.container_client.release_container(container_id)
        logger.debug(f"Released container {container_id} result {result}")
        return result

    async def utilize(self, resource_info, logger, **kwargs):  # type: ignore[override]
        container_id = resource_info["container_id"]

        logger.debug(f"Utilizing container {container_id} with {kwargs}")
        result = await self.container_client.execute(
            container_id,
            cwd=self.exec_params.get("cwd"),
            env=self.exec_params.get("env"),
            forward_env=self.exec_params.get("forward_env"),
            timeout=self.exec_params["timeout_secs"],
            **kwargs,
        )
        if "error" in result:
            logger.error(
                f"Error utilizing container {container_id}: {result['error']} with {kwargs}"
            )
            result = {"returncode": 1, "output": result["error"]}
        logger.debug(f"Utilized container {container_id}")
        return result

    async def __aenter__(self):
        await super().__aenter__()
        await self.container_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.container_client.__aexit__(exc_type, exc, tb)
        return await super().__aexit__(exc_type, exc, tb)

    def get_container_image(self, task: Dict[str, Any]) -> str:
        raise NotImplementedError("Please implement get_container_image method")

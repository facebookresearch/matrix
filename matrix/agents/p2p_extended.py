# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
import asyncio
import json
import logging
import os
import random
import re
import time
import traceback
from collections import defaultdict
from io import StringIO
from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple, Type, Union

os.environ.setdefault("OMP_NUM_THREADS", "1")
import ray
from datasets import load_dataset
from hydra.utils import get_class
from omegaconf import DictConfig, ListConfig, OmegaConf

from matrix.client import query_llm
from matrix.client.container_client import ContainerClient

from .agent_utils import setup_logging
from .p2p_agents import (
    AgentActor,
    BaseDatasetLoader,
    BaseMetricsAccumulator,
    BaseResourceClient,
    Orchestrator,
    Sink,
)

logger = logging.getLogger(__name__)


class RayDict(dict):
    """
    dict subclass for auto Ray storage/dereference of specific large text fields.
    Optimized for fixed fields: "text", "output".
    """

    FIXED_FIELDS = [
        "text",
        "output",  # in history
    ]
    TEXT_SIZE_THRESHOLD = 512  # default size threshold

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str) and key in self.FIXED_FIELDS:
            return super().__contains__(key) or super().__contains__(f"{key}_ref")
        return super().__contains__(key)

    async def get_async(self, key: str, default: Any = None) -> Any:
        if key in self.FIXED_FIELDS:
            ref_key = f"{key}_ref"
            if ref_key in self:
                return await super().__getitem__(ref_key)
        return super().get(key, default)

    @classmethod
    async def from_dict(cls, data: Dict[str, Any], registry: "Sink") -> "RayDict":
        self = cls()
        for k, v in data.items():
            if (
                k in cls.FIXED_FIELDS
                and isinstance(v, str)
                and len(v) > cls.TEXT_SIZE_THRESHOLD
            ):
                handle = ray.put(v)
                self[f"{k}_ref"] = handle
                await registry.register_object.remote([handle])  # type: ignore[attr-defined]
            elif k in cls.FIXED_FIELDS:
                # store small text directly but bypass the assert
                super(cls, self).__setitem__(k, v)
            else:
                self[k] = v
        return self

    async def to_dict(self) -> dict[str, Any]:
        out = dict(self)
        for key in self.FIXED_FIELDS:
            ref_key = f"{key}_ref"
            if ref_key in out:
                out[key] = await out[ref_key]
                del out[ref_key]
        return out

    async def cleanup_ray(self, sink: "Sink"):
        refs_to_free = [
            self[f"{field}_ref"]
            for field in self.FIXED_FIELDS
            if f"{field}_ref" in self and self[f"{field}_ref"] is not None
        ]
        if refs_to_free:
            await sink.unregister_object(refs_to_free)  # type: ignore[attr-defined]
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: ray.internal.free(refs_to_free))


class HistPair(NamedTuple):
    agent: str
    response: RayDict


class HistoryOrchestrator(Orchestrator):

    def __init__(self):
        super().__init__()
        # List of {"agent": str, "response": {"text": str, "tool_calls": [], "tool_call_id": id, "usage": {}, "extracted_answer": str, "status_ok": bool, "agreement": bool}}
        self.history: list[HistPair] = []

    async def update(
        self,
        result: Any,
        updater: AgentActor,
        logger: logging.Logger,
    ) -> "Orchestrator":
        """Update the orchestrator with the agent's result and determine the next agent."""
        if not isinstance(result, list):
            result = [result]
        for res in result:
            await self._append(self.current_agent(), res, updater.sink)  # type: ignore[arg-type]

        return self

    async def to_output(self) -> Dict[str, Any]:
        return {
            "current_agent": self.current_agent(),
            "id": self._id,
            "trial": self.trial,
            "seed": self.seed,
            "task": await self.get_task(),
            "history": [
                {"agent": msg.agent, "response": await msg.response.to_dict()}
                for msg in self.history
            ],
            "status": self.status,
        }

    async def init(
        self,
        simulation_id: str,
        first_agent: Tuple[Type, DictConfig],
        sink: "Sink",
        metadata: dict[str, Any],
        resources: dict[str, "BaseResourceClient"],
        logger: logging.Logger,
    ) -> None:
        await super().init(
            simulation_id, first_agent, sink, metadata, resources, logger
        )
        cls, agent_config = first_agent
        initial_message = await cls.get_task_message(agent_config, metadata["task"])  # type: ignore[attr-defined]
        if logger is not None:
            logger.debug(f"Get initial messageMetadataStart {self.id}")
        if initial_message is not None:
            await self._append(
                initial_message["agent"], initial_message["response"], sink
            )

    async def _append(self, agent: str, msg: dict[str, Any], sink: Sink):
        if "timestamp" not in msg:
            msg["timestamp"] = time.time()
        self.history.append(
            HistPair(agent=agent, response=await RayDict.from_dict(msg, sink))
        )

    async def cleanup(self, sink, resources, logger):
        await super().cleanup(sink, resources, logger)
        await asyncio.gather(
            *[hist.response.cleanup_ray(sink) for hist in self.history]
        )


class SequentialOrchestrator(HistoryOrchestrator):

    def __init__(
        self,
        interaction_order: List[str],
    ):
        super().__init__()
        self.interaction_order = interaction_order
        self._current_agent_index = 0

    def current_agent(self) -> str:
        if not self.interaction_order:
            raise ValueError("No interaction order defined")
        return self.interaction_order[self._current_agent_index]

    async def update(
        self,
        result: Any,
        updater: "AgentActor",
        logger: logging.Logger,
    ) -> "Orchestrator":
        """Update the orchestrator with the agent's result and determine the next agent."""
        await super().update(result, updater, logger)
        self._current_agent_index = (self._current_agent_index + 1) % len(
            self.interaction_order
        )
        return self


class HuggingfaceDatasetLoader(BaseDatasetLoader):
    def __init__(
        self,
        name: str,
        split: str,
        cut_off: Optional[int] = None,
        data_files: str | None = None,
        hub_download: bool = False,  # useful when pyarrow can't handle complex json
    ):
        super().__init__()
        self.dataset_name = name
        self.split = split
        if isinstance(data_files, str):
            data_files = os.path.expanduser(data_files)
        elif isinstance(data_files, list):
            data_files = [os.path.expanduser(f) for f in data_files]
        self.data_files = data_files
        self.cut_off = cut_off
        self.hub_download = hub_download
        self._count = 0
        self.done = False

    def load_data(self) -> Generator[Dict[str, Any], None, None]:
        if self.data_files is not None and self.hub_download:
            import json

            from huggingface_hub import hf_hub_download

            # Download the JSON file, automatically cached locally
            json_path = hf_hub_download(
                repo_id=self.dataset_name,
                filename=self.data_files,
                repo_type="dataset",
            )

            # Load it with Python json to get nested dictionary/list
            with open(json_path, "r") as f:
                dataset_val = json.load(f)
        else:
            dataset_val = load_dataset(
                self.dataset_name,
                split=self.split,
                data_files=self.data_files,
                streaming=True,
            )
        self._count = 0

        for item in dataset_val:
            if self.cut_off is not None and self._count >= self.cut_off:
                break
            item = self.transform(item)
            self._count += 1
            yield item

        self.done = True

    def transform(self, item) -> Dict[str, Any]:
        return dict(item)

    def total_count(self) -> Optional[int]:
        return self._count if self.done else None


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


class ContainerExecutionAgent(AgentActor):

    @abc.abstractmethod
    async def get_commands(
        self, orchestrator: HistoryOrchestrator
    ) -> List[Union[str, List[str]]]:
        pass

    async def preprocess(self, orchestrator: HistoryOrchestrator) -> Dict[str, Any]:  # type: ignore[override]
        return {"cmd": await self.get_commands(orchestrator)}

    async def process(self, orchestrator: Orchestrator, response: Any) -> Any:
        cmds = response["cmd"]
        results = []
        assert (
            self.resource_client is not None and orchestrator.resource_state is not None
        )
        for cmd in cmds:
            result = await self.resource_client.utilize(
                orchestrator.resource_state[self.resource_name],
                self.logger,
                cmd=cmd,
            )
            if "returncode" not in result:
                raise Exception(result)
            results.append(result)
        return {"results": results}


class LLMAgentActor(AgentActor):

    async def process(self, orchestrator: Orchestrator, response: Any) -> Any:
        assert self.resource_client is not None
        response = await self.resource_client.utilize(
            {},
            self.logger,
            messages=response,
            task_id=orchestrator.id,
            seed=orchestrator.seed,
        )
        return response

    async def postprocess(self, orchestrator: Orchestrator, response: Any) -> Any:  # type: ignore[override]
        if "text" in response and isinstance(response["text"], list):
            response["text"] = response["text"][0]
        if "tool_calls" in response:
            tool_calls = response["tool_calls"][0]  # because n == 1
            # because openai want a different input tool_call than the one it returns!
            for call in tool_calls:
                call["type"] = "function"
                call["function"] = {
                    "name": call["name"],
                    "arguments": call["arguments"],
                }
                call.pop("name")
                call.pop("arguments")
            response["tool_calls"] = tool_calls
        response["status_ok"] = "error" not in response
        return response

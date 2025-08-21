# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import glob
import json
import logging
import os
import random
import time
import typing as tp

import aiohttp
import tqdm
from fire import Fire

from matrix.client.client_utils import (
    get_an_endpoint_url,
    load_from_jsonl,
    save_to_jsonl,
)
from matrix.client.endpoint_cache import EndpointCache

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("process_vision_data")
logging.getLogger("httpx").setLevel(logging.WARNING)


class VisionClient:
    def __init__(
        self,
        url: tp.Union[str, tp.Callable[[], tp.Awaitable[str]]],
        timeout_secs: int = 120,
        limit: int = 0,
    ):
        """
        params:
            url: The base URL for the http endpoint
            timeout_secs: Per request timeout in seconds.
            limit: max num of inputs to take (0 means all inputs).
        """
        self.url = url
        self.timeout_secs = timeout_secs
        self.limit = limit

    def _validate_request(self, data: tp.Dict[str, tp.Any]) -> bool:
        """
        Validate the request data to ensure it contains the necessary fields.
        """
        keys_to_check = ("video", "image", "text")
        if not any(key in data for key in keys_to_check):
            logger.error(f"Request must contain at least one of {keys_to_check}.")
            return False
        if "video" in data:
            if "file_path" not in data["video"]:
                logger.error("Request must contain 'file_path' field in 'video'.")
                return False
        if "image" in data:
            if "file_path" not in data["image"]:
                logger.error("Request must contain 'file_path' field in 'image'.")
                return False
        return True

    async def make_request(
        self,
        data: tp.Dict[str, tp.Any],
        max_retries: int = 3,
        initial_delay: int = 1,
        backoff_factor: int = 2,
        endpoint_cache: tp.Optional[EndpointCache] = None,
    ) -> tp.Dict[str, tp.Any]:

        assert self._validate_request(data), "Invalid request data"

        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["request_timestamp"] = time.time()
        if "timeout" not in data:
            data["timeout"] = self.timeout_secs
        max_retries = max(1, max_retries)
        exception: tp.Optional[Exception] = None

        for attempt in range(max_retries):
            if callable(self.url):
                base_url = await self.url()
            elif not self.url and endpoint_cache:
                url = await get_an_endpoint_url(endpoint_cache, "")
                base_url = url
            else:
                base_url = self.url

            assert base_url.startswith("http")
            app_url = f"{base_url}/run"

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(app_url, json=data) as response:
                        status = response.status
                        content = await response.text()
                        result = {
                            "request": data,
                            "response": {
                                **json.loads(content),
                                "status": status,
                                "response_timestamp": time.time(),
                            },
                        }
                        return result
            except asyncio.TimeoutError as e:
                exception = e
                if attempt < max_retries - 1:
                    delay = initial_delay * (
                        backoff_factor**attempt + random.uniform(0, 1)
                    )
                    await asyncio.sleep(delay)
                    if endpoint_cache:
                        self.url = await get_an_endpoint_url(endpoint_cache, "", True)
            except Exception as e:
                exception = e
        return {
            "request": data,
            "response": {
                "error": str(exception or "unknown error"),
                "response_timestamp": time.time(),
            },
        }

    async def inference(
        self,
        output_file: str,
        input_jsonls: str,
        batch_size: int = 32,
    ):

        save_dir = os.path.dirname(output_file)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        if os.path.exists(output_file):
            logger.warning(
                f"Output file '{output_file}' already exists, overwriting..."
            )

        input_files = glob.glob(input_jsonls)
        if not input_files:
            logger.error(f"No input files found matching pattern: {input_jsonls}")
            return

        lines = load_from_jsonl(tuple(input_files))
        if self.limit <= 0:
            limit = len(lines)
        lines = lines[:limit]

        pbar = tqdm.tqdm(total=len(lines), desc="Send request")

        stats = {"success": 0, "total": 0, "sum_latency": 0}
        pending_tasks = set()  # type: ignore
        batch_results = []
        append_output_file = False

        async def save_outputs(flush=False):
            nonlocal pending_tasks, batch_results, append_output_file
            output_batch_size = 32

            if pending_tasks:
                completed, pending_tasks = await asyncio.wait(
                    pending_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for completed_task in completed:
                    batch_results.append(await completed_task)
                    pbar.update(1)
            if flush or len(batch_results) >= output_batch_size:
                await asyncio.to_thread(
                    save_to_jsonl,
                    batch_results,
                    output_file,
                    "w" if not append_output_file else "a",
                    stats,
                )
                batch_results = []
                append_output_file = True

        for line in lines:
            task = asyncio.create_task(
                self.make_request(
                    line,
                )
            )
            pending_tasks.add(task)
            if len(pending_tasks) >= batch_size:
                await save_outputs()
        while pending_tasks:
            await save_outputs()
        if batch_results:
            await save_outputs(flush=True)
        pbar.close()
        logger.info(f"Stats of the request: {stats}")


if __name__ == "__main__":
    Fire(VisionClient)

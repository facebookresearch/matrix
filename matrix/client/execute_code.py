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

from matrix.client.client_utils import get_an_endpoint_url, save_to_jsonl
from matrix.client.endpoint_cache import EndpointCache

# Configure logging for execute_code.py
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("execute_code")
# Optionally suppress noisy logs from imported modules if not already done
logging.getLogger("httpx").setLevel(logging.WARNING)


class CodeExcutionClient:
    def __init__(
        self,
        url: tp.Union[str, tp.Callable[[], tp.Awaitable[str]]],
        timeout_secs: int = 120,
        limit: int = 0,
    ):
        """
        Setup the code execution client.

        params:
            url: The base URL for the http endpoint (e.g., "http://localhost:8001/code").
            batch_size: Max number of concurrent requests.
            timeout_secs: Per request timeout in seconds.
            limit: max num of inputs to take (0 means all inputs).
        """
        self.url = url
        self.timeout_secs = timeout_secs
        self.limit = limit

    def load_from_jsonl(
        self,
        input_files: tp.Tuple[str, ...],
        text_keys: tp.Optional[list[str]] = None,
        prompt_template: tp.Optional[str] = None,
    ) -> tp.List[tp.Dict[str, tp.Any]]:

        def get_request(key: str, data: tp.Dict[str, tp.Any]) -> tp.Optional[tp.Any]:
            keys = key.split(".")
            current_data = data
            for k in keys:
                if isinstance(current_data, dict) and k in current_data:
                    current_data = current_data[k]
                else:
                    return None
            return current_data

        data = []
        for file_name in input_files:
            assert os.path.exists(file_name), f"{file_name} does not exist"
            with open(file_name, "r", encoding="UTF-8") as f:
                max_length = 0
                num_lines = 0
                for line_number, line in enumerate(f, start=1):
                    line_json = json.loads(line)
                    fields = {}
                    if text_keys is not None:
                        for text_key in text_keys:
                            text = get_request(text_key, line_json)
                            assert text is not None, f"Missing field {text_key}"
                            fields[text_key] = text
                    if prompt_template is not None:
                        prompt = prompt_template.format(**fields)
                    else:
                        prompt = ""
                    code = "\n".join(list(fields.values()) + [prompt])
                    item = {
                        "code": code,
                        "metadata": {"filename": file_name, "line": line_number},
                    }
                    max_length = max(len(item["code"]), max_length)
                    # Add metadata to the dictionary
                    data.append(item)
                    num_lines += 1
                print(
                    f"Loaded {num_lines} lines from {file_name}, max text length {max_length}"
                )
        return data

    async def make_request(
        self,
        data: tp.Dict[str, tp.Any],
        max_retries: int = 3,
        initial_delay: int = 1,
        backoff_factor: int = 2,
        endpoint_cache: tp.Optional[EndpointCache] = None,
    ) -> tp.Dict[str, tp.Any]:
        """
        Send request data to code execution app.

        params:
            data: request data, eg {"code": "assert 1 == 1"}
        """
        assert "code" in data, f"missing code field in {data}"
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
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(base_url, json=data) as response:
                        status = response.status  # Get the HTTP status code
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

    async def execute_code(
        self,
        output_file: str,
        input_jsonls: str,
        batch_size: int = 32,
        text_keys: tp.Optional[list[str]] = None,
        prompt_template: tp.Optional[str] = None,
    ):
        """
        Execute code prompts by sending requests to a backend via direct httpx.

        Reads code prompts from jsonl files, sends requests using the 'httpx' transport,
        and saves results to a jsonl file.

        params:
            output_file: name of the output jsonl file.
            input_jsonls: glob pattern for input jsonl files (e.g., "code_data/*.jsonl").
            text_keys: These columns are concated as `code` to send to backend.
            prompt_template: prompt template is appended to `code` after string interpolation,
                e.g. "check({entry_point})" will add check with entry_point column to `code`.
        """

        logger.info(
            f"Starting code execution with url: {self.url}, batch_size: {batch_size}, timeout: {self.timeout_secs}s"
        )

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

        lines = self.load_from_jsonl(
            tuple(input_files),
            text_keys,
            prompt_template,
        )
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
            # async with async_client.openai_client as client:
            task = asyncio.create_task(
                self.make_request(
                    line,
                )
            )
            pending_tasks.add(task)
            # If we have reached the batch size, wait for at least one task to complete
            if len(pending_tasks) >= batch_size:
                await save_outputs()
        while pending_tasks:
            await save_outputs()
        if batch_results:
            await save_outputs(flush=True)
        pbar.close()
        logger.info(f"Stats of the request: {stats}")


if __name__ == "__main__":
    Fire(CodeExcutionClient)

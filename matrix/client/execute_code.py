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


def load_from_jsonl(
    input_files: tp.Tuple[str, ...],
    text_keys: list[str],
    prompt_template: str,
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
                for text_key in text_keys:
                    text = get_request(text_key, line_json)
                    assert text is not None, f"Missing field {text_key}"
                    fields[text_key] = text
                prompt = prompt_template.format(**fields)
                code = "\n".join(list(fields.values()) + [prompt])
                item = {
                    "code": code,
                    "metadata": {"filename": file_name, "line": line_number},
                }
                max_length = max(len(item["code"]), max_length)
                # Add metadata to the dictionary
                data.append(item)
            logger.info(
                f"Loaded {num_lines} lines from {file_name}, max text length {max_length}"
            )
    return data


async def make_request(
    url: tp.Union[str, tp.Callable[[], tp.Awaitable[str]]],
    data: tp.Dict[str, tp.Any],
    max_retries: int = 3,
    initial_delay: int = 1,
    backoff_factor: int = 2,
    timeout_secs: int = 600,
    endpoint_cache: tp.Optional[EndpointCache] = None,
) -> tp.Dict[str, tp.Any]:
    if "metadata" not in data:
        data["metadata"] = {}
    data["metadata"]["request_timestamp"] = time.time()
    max_retries = max(1, max_retries)
    exception: tp.Optional[Exception] = None

    for attempt in range(max_retries):
        if callable(url):
            base_url = await url()
        elif not url and endpoint_cache:
            url = await get_an_endpoint_url(endpoint_cache, "")
            base_url = url
        else:
            base_url = url

        assert base_url.startswith("http")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    base_url, json={"code": data["code"], "timeout": timeout_secs}
                ) as response:
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
                delay = initial_delay * (backoff_factor**attempt + random.uniform(0, 1))
                await asyncio.sleep(delay)
                if endpoint_cache:
                    url = await get_an_endpoint_url(endpoint_cache, "", True)
        except Exception as e:
            exception = e
    return {
        "request": data,
        "response": {
            "error": str(exception or "unknown error"),
            "response_timestamp": time.time(),
        },
    }


async def main(
    url: tp.Union[str, tp.Callable[[], tp.Awaitable[str]]],
    output_file: str,
    input_jsonls: str,
    text_keys: list[str],  # Key for the code prompt in input jsonl
    prompt_template: str = "check({entry_point})",
    batch_size: int = 32,
    timeout_secs: int = 120,  # Shorter timeout might be appropriate for execution
    limit: int = 0,
):
    """
    Execute code prompts by sending requests to a backend via direct httpx.

    Reads code prompts from jsonl files, sends requests using the 'httpx' transport,
    and saves results to a jsonl file.

    params:
        url: The base URL for the httpx endpoint (e.g., "http://localhost:8001").
        output_file: name of the output jsonl file.
        input_jsonls: glob pattern for input jsonl files (e.g., "code_data/*.jsonl").
        batch_size: Max number of concurrent requests.
        text_keys: The keys for raw text code prompts in input jsonl.
        prompt_template: prompt template, e.g. "check({entry_point})".
        timeout_secs: Per request timeout in seconds.
    """

    logger.info(
        f"Starting code execution with url: {url}, batch_size: {batch_size}, timeout: {timeout_secs}s"
    )

    save_dir = os.path.dirname(output_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(output_file):
        logger.warning(f"Output file '{output_file}' already exists, overwriting...")

    input_files = glob.glob(input_jsonls)
    if not input_files:
        logger.error(f"No input files found matching pattern: {input_jsonls}")
        return

    lines = load_from_jsonl(
        tuple(input_files),
        text_keys,
        prompt_template,
    )
    if limit <= 0:
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
            make_request(
                url,
                line,
                timeout_secs=timeout_secs,
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
    # Fire runs the main function with command line arguments
    Fire(main)

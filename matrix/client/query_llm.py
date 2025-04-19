# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import glob
import hashlib
import json
import logging
import os
import random
import time
import typing as tp
from functools import reduce

import grpc
import httpx
import openai
import tqdm
from fire import Fire
from google.protobuf import json_format
from grpc import aio as grpc_aio
from openai import APIConnectionError, APITimeoutError, RateLimitError

from matrix.app_server.llm import openai_pb2, openai_pb2_grpc
from matrix.client.endpoint_cache import EndpointCache

CHAR_PER_TOKEN = 3.61
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("query_llm")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)


def convert_llama_instruct_text(text: str) -> tp.List[tp.Dict[str, str]]:
    messages = []
    start_header_id = "<|start_header_id|>"
    end_header_id = "<|end_header_id|>"
    eot_id = "<|eot_id|>"
    while start_header_id in text:
        start_index = text.find(start_header_id)
        end_index = text.find(end_header_id) + len(end_header_id)
        role = text[start_index + len(start_header_id) : end_index - len(end_header_id)]

        next_start_index = text.find(eot_id, end_index) + len(eot_id)
        content = text[end_index : (next_start_index - len(eot_id))].strip()
        messages.append({"role": role, "content": content})
        text = text[next_start_index:]
    if not messages:
        # no roles
        messages.append({"role": "user", "content": text})
    if messages[-1]["role"] == "assistant":
        messages = messages[:-1]
    return messages


def load_from_jsonl(
    input_files: tp.Tuple[str, ...],
    text_key: str,
    messages_key: str,
    system_prompt: str,
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

    def get_metadata_key(text_key: str) -> str:
        parts = text_key.split(".")
        parts[-1] = "metadata"
        return ".".join(parts)

    def load_json_line(
        file_name: str, line: str, line_number: int, system_prompt: str
    ) -> tp.Dict[str, tp.Any]:
        try:
            data = json.loads(line)
            text = get_request(text_key, data)
            if text:
                messages = convert_llama_instruct_text(text)
                metadata = get_request(get_metadata_key(text_key), data)
            else:
                messages = get_request(messages_key, data)  # type: ignore
                assert messages, "either {text_key} or {messages_key} should exist"
                metadata = get_request(get_metadata_key(messages_key), data)

            if system_prompt:
                if messages[0]["role"] == "system":
                    messages[0]["content"] = system_prompt
                else:
                    messages.insert(0, {"role": "system", "content": system_prompt})

            if metadata is None:
                metadata = {"filename": file_name, "line": line_number}
            return {
                "metadata": metadata,
                "messages": messages,
            }
        except Exception as e:
            raise ValueError(f"Error in line {line_number}\n{line} of {file_name}: {e}")

    def get_text_length(messages: tp.List[tp.Dict[str, str]]) -> int:
        return reduce(lambda x, y: x + y, [len(m["content"]) for m in messages])

    data = []
    for file_name in input_files:
        assert os.path.exists(file_name), f"{file_name} does not exist"
        with open(file_name, "r", encoding="UTF-8") as f:
            max_length = 0
            num_lines = 0
            for num_lines, line in enumerate(f, start=1):
                item = load_json_line(file_name, line, num_lines, system_prompt)
                max_length = max(get_text_length(item["messages"]), max_length)
                # Add metadata to the dictionary
                data.append(item)
            logger.info(
                f"Loaded {num_lines} lines from {file_name}, max text length {max_length}, estimated max token {int(max_length / CHAR_PER_TOKEN)}"
            )
    return data


def save_to_jsonl(
    data: tp.List[tp.Dict[str, tp.Any]],
    filename: str,
    write_mode: str,
    stats: tp.Dict[str, tp.Any],
) -> None:
    with open(filename, write_mode) as file:
        for item in data:
            stats["total"] += 1
            stats["success"] += 0 if "error" in item["response"] else 1
            stats["sum_latency"] += (
                item["response"]["response_timestamp"]
                - item["request"]["metadata"]["request_timestamp"]
            )
            json_str = json.dumps(item)
            file.write(json_str + "\n")


async def get_an_endpoint_url(
    endpoint_cache: EndpointCache,
    multiplexed_model_id: str = "",
    force_update: bool = False,
) -> str:
    urls = await endpoint_cache(force_update)
    start_time = time.time()
    while not urls:
        # explicitly use synchronous sleep to block the whole event loop
        time.sleep(60)
        print(f"no worker is available, waited {time.time() - start_time}s..")
        urls = await endpoint_cache(force_update)

    if multiplexed_model_id:
        hashed_int = int(hashlib.sha256(multiplexed_model_id.encode()).hexdigest(), 16)
        return urls[hashed_int % len(urls)]
    else:
        return random.choice(urls)


def _convert_token_log_probs(token_log_probs):
    if not token_log_probs.token_map:
        return None
    result = {}
    for key, value in token_log_probs.token_map.items():
        result[str(key)] = {
            "logprob": value.logprob,
            "rank": value.rank,
            "decoded_token": value.decoded_token,
        }
    return result


async def make_request(
    url: tp.Union[str, tp.Callable[[], tp.Awaitable[str]]],
    model: str,
    data: tp.Dict[str, tp.Any],
    seed: int = 42,
    app_name: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 0.9,
    n: int = 1,
    logprobs: bool = False,
    max_retries: int = 3,
    initial_delay: int = 1,
    backoff_factor: int = 2,
    multiplexed_model_id: str = "",
    timeout_secs: int = 600,
    prompt_logprobs: tp.Optional[int] = None,
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
            url = await get_an_endpoint_url(endpoint_cache, multiplexed_model_id)
            base_url = url
        else:
            base_url = url

        if base_url.startswith("http"):
            async with openai.AsyncOpenAI(
                base_url=base_url,
                api_key="EMPTY",  # Use your API key
                max_retries=3,
            ) as client:
                try:
                    if "messages" in data:
                        response = await client.chat.completions.create(
                            model=model,
                            messages=data["messages"],
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            seed=seed,
                            n=n,
                            timeout=timeout_secs,  # 10 minutes
                            logprobs=logprobs,
                            extra_headers=(
                                {"serve_multiplexed_model_id": multiplexed_model_id}
                            ),
                        )
                        result = {
                            "request": data,
                            "response": {
                                "text": [
                                    response.choices[i].message.content
                                    for i in range(n)
                                ],
                                "finish_reason": [
                                    response.choices[i].finish_reason for i in range(n)
                                ],
                                "response_timestamp": time.time(),
                            },
                        }
                        if logprobs and response.choices[0].logprobs is not None:
                            lp = [
                                [
                                    {"token": elem.token, "logprob": elem.logprob}
                                    for elem in response.choices[i].logprobs.content  # type: ignore[union-attr]
                                ]
                                for i in range(n)
                            ]
                            result["response"]["logprobs"] = lp
                    elif "prompt" in data:
                        response = await client.completions.create(  # type: ignore[assignment]
                            model=model,
                            prompt=data["prompt"],
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            seed=seed,
                            n=n,
                            timeout=timeout_secs,
                            logprobs=logprobs,
                            extra_headers=(
                                {"serve_multiplexed_model_id": multiplexed_model_id}
                            ),
                            extra_body={
                                "prompt_logprobs": prompt_logprobs,
                            },
                        )
                        result = {
                            "request": data,
                            "response": {
                                "text": [response.choices[i].text for i in range(n)],  # type: ignore[attr-defined]
                                "finish_reason": [
                                    response.choices[i].finish_reason for i in range(n)
                                ],
                                "response_timestamp": time.time(),
                            },
                        }
                        if logprobs and response.choices[0].logprobs is not None:  # type: ignore[attr-defined]
                            lp = [
                                [
                                    {"token": elem[0], "logprob": elem[1]}
                                    for elem in zip(
                                        response.choices[i].logprobs.tokens,  # type: ignore[union-attr]
                                        response.choices[i].logprobs.token_logprobs,  # type: ignore[union-attr]
                                    )  # type: ignore[attr-defined]
                                ]
                                for i in range(n)
                            ]
                            result["response"]["logprobs"] = lp
                        if (
                            prompt_logprobs is not None
                            and response.choices[0].prompt_logprobs is not None  # type: ignore[attr-defined]
                        ):
                            lp = [response.choices[i].prompt_logprobs for i in range(n)]  # type: ignore[attr-defined]
                            result["response"]["prompt_logprobs"] = lp
                    else:
                        raise Exception(
                            "request data should either have 'messeages' or 'prompt'!"
                        )
                    if response.usage:
                        result["response"]["usage"] = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                        }
                    return result
                except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                    exception = e
                    if attempt < max_retries - 1:
                        delay = initial_delay * (
                            backoff_factor**attempt + random.uniform(0, 1)
                        )
                        await asyncio.sleep(delay)
                        if endpoint_cache:
                            url = await get_an_endpoint_url(
                                endpoint_cache, multiplexed_model_id, True
                            )
                except Exception as e:
                    exception = e
        else:
            # it is grpc
            assert app_name, "app_name is required for grpc"
            async with grpc.aio.insecure_channel(base_url) as channel:
                try:
                    stub = openai_pb2_grpc.OpenaiServiceStub(channel)
                    metadata = (
                        ("application", app_name),
                        ("multiplexed_model_id", multiplexed_model_id),
                    )  # add multiplexed_model_id https://docs.ray.io/en/latest/serve/advanced-guides/grpc-guide.html

                    if "messages" in data:
                        messages = [
                            openai_pb2.CompletionMessage(  # type: ignore[attr-defined]
                                role=msg["role"], content=msg["content"]
                            )
                            for msg in data["messages"]
                        ]
                        request = openai_pb2.ChatCompletionRequest(  # type: ignore[attr-defined]
                            model=model,
                            messages=messages,
                            top_p=top_p,
                            temperature=temperature,
                            n=n,
                            seed=seed,
                            max_tokens=max_tokens,
                            logprobs=logprobs,
                        )
                        response = await stub.CreateChatCompletion(
                            request=request, metadata=metadata, timeout=timeout_secs
                        )
                        result = {
                            "request": data,
                            "response": {
                                "text": [response.choices[i].message.content for i in range(n)],  # type: ignore[attr-defined]
                                "finish_reason": [response.choices[i].finish_reason for i in range(n)],  # type: ignore[attr-defined]
                                "response_timestamp": time.time(),
                            },
                        }
                        if logprobs and response.choices[0].logprobs is not None:  # type: ignore[attr-defined]
                            lp = [
                                [
                                    {"token": elem.token, "logprob": elem.logprob}
                                    for elem in response.choices[i].logprobs.content  # type: ignore[union-attr]
                                ]
                                for i in range(n)
                            ]
                            result["response"]["logprobs"] = lp
                    elif "prompt" in data:
                        request = openai_pb2.CompletionRequest(  # type: ignore[attr-defined]
                            model=model,
                            prompt=data["prompt"],
                            top_p=top_p,
                            temperature=temperature,
                            n=n,
                            seed=seed,
                            max_tokens=max_tokens,
                            logprobs=logprobs,
                            prompt_logprobs=prompt_logprobs,
                        )
                        response = await stub.CreateCompletion(
                            request=request, metadata=metadata, timeout=timeout_secs
                        )
                        result = {
                            "request": data,
                            "response": {
                                "text": [response.choices[i].text for i in range(n)],  # type: ignore[attr-defined]
                                "finish_reason": [response.choices[i].finish_reason for i in range(n)],  # type: ignore[attr-defined]
                                "response_timestamp": time.time(),
                            },
                        }
                        if logprobs and response.choices[0].logprobs is not None:  # type: ignore[attr-defined]
                            lp = [
                                [
                                    {"token": elem[0], "logprob": elem[1]}
                                    for elem in zip(
                                        response.choices[i].logprobs.tokens,  # type: ignore[union-attr]
                                        response.choices[i].logprobs.token_logprobs,  # type: ignore[union-attr]
                                    )
                                ]
                                for i in range(n)
                            ]
                            result["response"]["logprobs"] = lp
                        if prompt_logprobs and response.choices[0].prompt_logprobs is not None:  # type: ignore[attr-defined]
                            lp = [
                                [
                                    _convert_token_log_probs(elem)
                                    for elem in response.choices[i].prompt_logprobs  # type: ignore[attr-defined]
                                ]
                                for i in range(n)
                            ]
                            result["response"]["prompt_logprobs"] = lp
                    else:
                        raise Exception(
                            "request data should either have 'messeages' or 'prompt'!"
                        )

                    if response.usage is not None:  # type: ignore[attr-defined]
                        result["response"]["usage"] = {
                            "prompt_tokens": response.usage.prompt_tokens,  # type: ignore[attr-defined]
                            "completion_tokens": response.usage.completion_tokens,  # type: ignore[attr-defined]
                        }
                    return result
                except grpc_aio.AioRpcError as e:
                    exception = e
                    status_code = e.code()
                    if status_code in [
                        grpc.StatusCode.DEADLINE_EXCEEDED,
                        grpc.StatusCode.UNAVAILABLE,
                        grpc.StatusCode.RESOURCE_EXHAUSTED,
                    ]:
                        if attempt < max_retries - 1:
                            delay = initial_delay * (
                                backoff_factor**attempt + random.uniform(0, 1)
                            )
                            await asyncio.sleep(delay)
                            # force to get a new url
                            if endpoint_cache:
                                url = await get_an_endpoint_url(
                                    endpoint_cache, multiplexed_model_id, True
                                )

                except asyncio.TimeoutError as e:
                    exception = e
                    if attempt < max_retries - 1:
                        delay = initial_delay * (
                            backoff_factor**attempt + random.uniform(0, 1)
                        )
                        await asyncio.sleep(delay)
                        if endpoint_cache:
                            url = await get_an_endpoint_url(
                                endpoint_cache, multiplexed_model_id, True
                            )
                except Exception as e:
                    exception = e
    return {
        "request": data,
        "response": {
            "error": str(exception or "unknown error"),
            "response_timestamp": time.time(),
        },
    }


def batch_requests(
    url: tp.Union[str, tp.Callable[[], tp.Awaitable[str]]],
    model: str,
    requests: tp.List[tp.Dict[str, tp.Any]],
    **kwargs,
) -> tp.List[tp.Dict[str, tp.Any]]:
    """
    Process multiple requests by calling make_request for each.
    This function works whether called from sync or async context.
    """

    async def _process_requests():
        """Helper function to process all requests concurrently."""
        return await asyncio.gather(
            *[make_request(url, model, request, **kwargs) for request in requests]
        )

    # Get the event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No event loop in this thread, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Check if we're already in an async context
    if loop.is_running():
        # We're in an async context and can't use run_until_complete
        # Create a new thread to run our async code
        import concurrent.futures
        import threading

        def run_in_new_loop():
            # Create a new event loop for this thread
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(_process_requests())
            finally:
                new_loop.close()

        # Run in an executor to avoid blocking the current event loop
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(run_in_new_loop).result()
    else:
        # We're in a sync context, use the current loop
        return loop.run_until_complete(_process_requests())


async def main(
    url: tp.Union[str, tp.Callable[[], tp.Awaitable[str]]],
    output_file: str,
    input_jsonls: str,
    app_name="",
    model="meta-llama/Meta-Llama-3.1-405B-Instruct",
    batch_size=32,
    seed=42,
    temperature=0.7,
    max_tokens=4096,
    top_p=0.9,
    n=1,
    logprobs: bool = False,
    text_key="text",
    messages_key="request.messages",
    system_prompt="",
    timeout_secs=600,
    override_output_file: bool = False,
    append_output_file: bool = False,
) -> tp.Dict[str, int]:
    """Send jsonl llama3 instruct prompt for inference and save both the request and response as jsonl.
    params:
    url: Llama openai endpoint, eg http://hostname:8000/405B/v1
    output_file: name of the output jsonl file.
    input_jsonls: variable num of input jsonl files, each line is a json with two formats
        1. {text_key: prompt} if text_key is found, prompt is raw text
        2. {messages_key: Iterable[ChatCompletionMessageParam]} if messages_key is found.
    model: the huggingface model name or a directory.
    batch_size: max number of concurrent requests.
    seed: seed.
    temperature: temperature for decoding.
    max_tokens: max num of output tokens.
    top_p: top_p for necleus sampling.
    text_key: the text field in the input json.
    messages_key: the messages field in the input json.
    system_prompt: system prompt to use.
    timeout_secs: per request timeout in seconds.
    override_output_file: Override given output file if it exists.
    append_output_file: Append to given output file if it exists.

    Return
    stats of the inference.
    """

    logger.info(
        f"url: {url}, batch_size: {batch_size}, temperature: {temperature}, max_tokens: {max_tokens}, top_p: {top_p}, seed: {seed}"
    )

    save_dir = os.path.dirname(output_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if override_output_file and append_output_file:
        raise ValueError(
            "Conflicting output file options. Please choose either `override_output_file` or `append_output_file`, but not both."
        )
    if os.path.exists(output_file) and not (override_output_file or append_output_file):
        raise FileExistsError(
            f"Output file '{output_file}' already exists. To proceed, please remove the file, set `override_output_file=True` to overwrite it, or set `append_output_file=True` to append to it."
        )
    lines = load_from_jsonl(
        tuple(glob.glob(input_jsonls)),
        text_key,
        messages_key,
        system_prompt=system_prompt,
    )
    pbar = tqdm.tqdm(total=len(lines), desc="Send request")

    stats = {"success": 0, "total": 0, "sum_latency": 0}
    pending_tasks = set()  # type: ignore
    batch_results = []

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
                model,
                line,
                app_name=app_name,
                seed=seed,
                top_p=top_p,
                n=n,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=logprobs,
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
    return stats


if __name__ == "__main__":
    Fire(main)

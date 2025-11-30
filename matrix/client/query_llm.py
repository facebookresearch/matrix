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
import uuid
from functools import partial, reduce

import grpc
import openai
import tqdm
from fire import Fire
from grpc import aio as grpc_aio
from openai import APIConnectionError, APITimeoutError, RateLimitError
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.chat_completion_tool_union_param import (
    ChatCompletionToolUnionParam,
)

from matrix.app_server.llm import openai_pb2, openai_pb2_grpc
from matrix.client.client_utils import get_an_endpoint_url, save_to_jsonl
from matrix.client.endpoint_cache import EndpointCache
from matrix.utils.os import batch_requests_async, run_async

CHAR_PER_TOKEN = 3.61
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("query_llm")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)


def convert_llama_instruct_text(
    text: str, keep_assistant=False
) -> tp.List[tp.Dict[str, str]]:
    messages = []
    start_header_id = "<|start_header_id|>"
    end_header_id = "<|end_header_id|>"
    eot_id = "<|eot_id|>"
    while start_header_id in text:
        start_index = text.find(start_header_id)
        end_index = text.find(end_header_id)
        role = text[start_index + len(start_header_id) : end_index]
        end_index += len(end_header_id)
        next_start_index = text.find(eot_id, end_index)
        if next_start_index == -1:
            next_start_index = len(text)
        content = text[end_index:next_start_index].lstrip()
        next_start_index += len(eot_id)
        messages.append({"role": role, "content": content})
        text = text[next_start_index:]
    if not messages:
        # no roles
        messages.append({"role": "user", "content": text})
    if not keep_assistant and messages[-1]["role"] == "assistant":
        assert not messages[-1][
            "content"
        ], "Last message in chat should not be assistant."
        messages = messages[:-1]
    return messages


def _get_request(key: str, data: tp.Dict[str, tp.Any]) -> tp.Optional[tp.Any]:
    keys = key.split(".")
    current_data = data
    for k in keys:
        if isinstance(current_data, dict) and k in current_data:
            current_data = current_data[k]
        else:
            return None
    return current_data


def _get_metadata_key(text_key: str) -> str:
    parts = text_key.split(".")
    parts[-1] = "metadata"
    return ".".join(parts)


def _prepare_request(
    sample: tp.Dict[str, tp.Any],
    text_key: str,
    messages_key: str,
    system_prompt: str,
    default_metadata: tp.Dict[str, tp.Any],
) -> tp.Dict[str, tp.Any]:
    text = _get_request(text_key, sample)
    if text:
        messages = convert_llama_instruct_text(text)
        metadata = _get_request(_get_metadata_key(text_key), sample)
    else:
        messages = _get_request(messages_key, sample)  # type: ignore
        assert messages, f"either {text_key} or {messages_key} should exist"
        metadata = _get_request(_get_metadata_key(messages_key), sample)

    if system_prompt:
        if messages[0]["role"] == "system":
            messages[0]["content"] = system_prompt
        else:
            messages.insert(0, {"role": "system", "content": system_prompt})

    if metadata is None:
        metadata = default_metadata
    return {"metadata": metadata, "messages": messages}


def load_from_jsonl(
    input_files: tp.Tuple[str, ...],
    text_key: str,
    messages_key: str,
    system_prompt: str,
) -> tp.List[tp.Dict[str, tp.Any]]:

    def load_json_line(
        file_name: str, line: str, line_number: int
    ) -> tp.Dict[str, tp.Any]:
        try:
            data = json.loads(line)
            return _prepare_request(
                data,
                text_key,
                messages_key,
                system_prompt,
                {"filename": file_name, "line": line_number},
            )
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
                item = load_json_line(file_name, line, num_lines)
                max_length = max(get_text_length(item["messages"]), max_length)
                # Add metadata to the dictionary
                data.append(item)
            logger.info(
                f"Loaded {num_lines} lines from {file_name}, max text length {max_length}, estimated max token {int(max_length / CHAR_PER_TOKEN)}"
            )
    return data


def load_from_hf_dataset(
    dataset_name: str,
    split: str,
    text_key: str,
    messages_key: str,
    system_prompt: str,
) -> tp.List[tp.Dict[str, tp.Any]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    data = []
    for idx, sample in enumerate(dataset):
        data.append(
            _prepare_request(
                sample,
                text_key,
                messages_key,
                system_prompt,
                {"index": idx},
            )
        )
    logger.info(f"Loaded {len(data)} samples from {dataset_name} split {split}")
    return data


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


def make_error_response(
    request: tp.Dict[str, tp.Any], exception: Exception | None
) -> tp.Dict[str, tp.Any]:
    return {
        "request": request,
        "response": {
            "error": str(exception or "unknown error"),
            "response_timestamp": time.time(),
        },
    }


async def make_request(
    url: tp.Union[None, str, tp.Callable[[], tp.Awaitable[str]]],
    model: str,
    data: tp.Dict[str, tp.Any],
    seed: tp.Optional[int] = None,
    app_name: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 0.9,
    n: int = 1,
    logprobs: bool = False,
    top_logprobs: tp.Optional[int] = None,
    max_retries: int = 3,
    initial_delay: int = 1,
    backoff_factor: int = 2,
    multiplexed_model_id: str = "",
    timeout_secs: int = 600,
    tools: tp.Iterable[ChatCompletionToolUnionParam] | NotGiven = NOT_GIVEN,
    tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
    prompt_logprobs: tp.Optional[int] = None,
    endpoint_cache: tp.Optional[EndpointCache] = None,
    top_k: int = -1,
    guided_decoding: tp.Optional[tp.Dict[str, tp.Any]] = None,
    extra_body: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> tp.Dict[str, tp.Any]:
    if "metadata" not in data:
        data["metadata"] = {}
    data["metadata"]["request_timestamp"] = time.time()
    max_retries = max(1, max_retries)
    exception: tp.Optional[Exception] = None

    extra_body = extra_body or {}
    if top_k != -1:
        extra_body["top_k"] = top_k
    if guided_decoding:
        if "json" in guided_decoding:
            extra_body["guided_json"] = guided_decoding["json"]
        if "regex" in guided_decoding:
            extra_body["guided_regex"] = guided_decoding["regex"]
        if "choice" in guided_decoding:
            extra_body["guided_choice"] = guided_decoding["choice"]
        if "grammar" in guided_decoding:
            extra_body["guided_grammar"] = guided_decoding["grammar"]
    if prompt_logprobs:
        extra_body["prompt_logprobs"] = prompt_logprobs
    if len(extra_body) == 0:
        extra_body = None  # type: ignore[assignment]
    if multiplexed_model_id:
        extra_headers = {"serve_multiplexed_model_id": multiplexed_model_id}
    else:
        extra_headers = None
    for attempt in range(max_retries):
        if callable(url):
            base_url = await url()
        elif not url and endpoint_cache:
            url = await get_an_endpoint_url(endpoint_cache, multiplexed_model_id)
            base_url = url
        else:
            assert url is not None
            base_url = url

        if base_url.startswith("http"):
            async with openai.AsyncOpenAI(
                base_url=base_url,
                api_key="EMPTY",  # Use your API key
                max_retries=3,
            ) as client:
                try:
                    if "messages" in data:
                        response = await client.chat.completions.create(  # type: ignore[misc]
                            model=model,
                            messages=data["messages"],
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            seed=seed,
                            n=n,
                            timeout=timeout_secs,  # 10 minutes
                            tools=tools,  # type: ignore[arg-type]
                            tool_choice=tool_choice,  # type: ignore[arg-type]
                            logprobs=logprobs or top_logprobs is not None,
                            top_logprobs=top_logprobs,
                            extra_headers=extra_headers,
                            extra_body=extra_body,
                        )
                        result = {
                            "request": data,
                            "response": {
                                "finish_reason": [
                                    response.choices[i].finish_reason for i in range(n)
                                ],
                                "response_timestamp": time.time(),
                            },
                        }
                        message0 = response.choices[0].message
                        if message0.content:
                            result["response"]["text"] = [response.choices[i].message.content for i in range(n)]  # type: ignore[attr-defined]
                        if (
                            hasattr(message0, "reasoning_content")
                            and message0.reasoning_content
                        ):
                            result["response"]["reasoning_content"] = [response.choices[i].message.reasoning_content for i in range(n)]  # type: ignore[attr-defined]
                        if message0.tool_calls:
                            result["response"]["tool_calls"] = [
                                [
                                    {
                                        "name": tool_call.function.name,  # type: ignore[union-attr]
                                        "arguments": tool_call.function.arguments,  # type: ignore[union-attr]
                                        "id": tool_call.id or str(uuid.uuid4()),
                                    }
                                    for tool_call in response.choices[
                                        i
                                    ].message.tool_calls  # type: ignore[union-attr]
                                ]
                                for i in range(n)
                            ]
                        if (logprobs or top_logprobs is not None) and response.choices[
                            0
                        ].logprobs is not None:
                            lp = [
                                [
                                    {
                                        "token": elem.token,
                                        "logprob": elem.logprob,
                                        **(
                                            {
                                                "top_logprobs": [
                                                    {
                                                        "token": tl.token,
                                                        "logprob": tl.logprob,
                                                    }
                                                    for tl in elem.top_logprobs
                                                ]
                                            }
                                            if elem.top_logprobs
                                            else {}
                                        ),
                                    }
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
                            logprobs=(
                                top_logprobs
                                if top_logprobs is not None
                                else (1 if logprobs else None)
                            ),
                            extra_headers=extra_headers,
                            extra_body=extra_body,
                        )
                        result = {
                            "request": data,
                            "response": {
                                "text": [response.choices[i].text for i in range(n)],  # type: ignore[attr-defined]
                                "finish_reason": [response.choices[i].finish_reason for i in range(n)],  # type: ignore[attr-defined]
                                "response_timestamp": time.time(),
                            },
                        }
                        if (logprobs or top_logprobs is not None) and response.choices[
                            0
                        ].logprobs is not None:  # type: ignore[attr-defined]
                            lp = [
                                [
                                    {
                                        "token": token,
                                        "logprob": lp_token,
                                        **(
                                            {
                                                "top_logprobs": [
                                                    {"token": t, "logprob": lp}
                                                    for t, lp in top_lp.items()
                                                ]
                                            }
                                            if top_lp
                                            else {}
                                        ),
                                    }
                                    for token, lp_token, top_lp in zip(
                                        response.choices[i].logprobs.tokens,  # type: ignore[union-attr]
                                        response.choices[i].logprobs.token_logprobs,  # type: ignore[union-attr]
                                        response.choices[i].logprobs.top_logprobs  # type: ignore[union-attr]
                                        or [
                                            None
                                            for _ in range(
                                                len(response.choices[i].logprobs.tokens)  # type: ignore[union-attr]
                                            )
                                        ],
                                    )
                                ]
                                for i in range(n)
                            ]
                            result["response"]["logprobs"] = lp
                        if prompt_logprobs is not None and response.choices[0].prompt_logprobs is not None:  # type: ignore[attr-defined]
                            # note: only grpc need conversion
                            lp = [response.choices[i].prompt_logprobs for i in range(n)]  # type: ignore[attr-defined]
                            result["response"]["prompt_logprobs"] = lp
                    else:
                        raise Exception(
                            "request data should either have 'messages' or 'prompt'!"
                        )
                    if response.usage:
                        result["response"]["usage"] = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            **(
                                {"total_tokens": response.usage.total_tokens}
                                if getattr(response.usage, "total_tokens", None)
                                is not None
                                else {}
                            ),
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
            assert (
                top_k == -1 and not guided_decoding
            ), "top_k and guided_decoding are not supported for grpc"
            async with grpc.aio.insecure_channel(base_url) as channel:
                try:
                    stub = openai_pb2_grpc.OpenaiServiceStub(channel)
                    metadata = (
                        ("application", app_name),
                        ("multiplexed_model_id", multiplexed_model_id),
                    )  # add multiplexed_model_id https://docs.ray.io/en/latest/serve/advanced-guides/grpc-guide.html

                    if "messages" in data:
                        messages = []
                        for msg in data["messages"]:
                            tool_calls = []
                            for tc in msg.get("tool_calls", []):
                                tool_calls.append(
                                    openai_pb2.ToolCall(  # type: ignore[attr-defined]
                                        id=tc["id"],
                                        type="function",
                                        function=openai_pb2.FunctionCall(  # type: ignore[attr-defined]
                                            name=tc["function"]["name"],
                                            arguments=tc["function"].get(
                                                "arguments", ""
                                            ),
                                        ),
                                    )
                                )
                            msg_proto = openai_pb2.CompletionMessage(  # type: ignore[attr-defined]
                                role=msg["role"],
                                content=msg.get("content"),
                                tool_calls=tool_calls or None,
                                tool_call_id=msg.get("tool_call_id"),
                            )
                            messages.append(msg_proto)
                        request = openai_pb2.ChatCompletionRequest(  # type: ignore[attr-defined]
                            model=model,
                            messages=messages,
                            top_p=top_p,
                            temperature=temperature,
                            tool_choice=(
                                tool_choice if tool_choice is not NOT_GIVEN else None
                            ),
                            tools=tools if tools is not NOT_GIVEN else None,
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
                                "finish_reason": [response.choices[i].finish_reason for i in range(n)],  # type: ignore[attr-defined]
                                "response_timestamp": time.time(),
                            },
                        }
                        message0 = response.choices[0].message
                        if message0.content:
                            result["response"]["text"] = [response.choices[i].message.content for i in range(n)]  # type: ignore[attr-defined]
                        if (
                            hasattr(message0, "reasoning_content")
                            and message0.reasoning_content
                        ):
                            result["response"]["reasoning_content"] = [response.choices[i].message.reasoning_content for i in range(n)]  # type: ignore[attr-defined]
                        if message0.tool_calls:
                            result["response"]["tool_calls"] = [
                                [
                                    {
                                        "name": tool_call.function.name,  # type: ignore[union-attr]
                                        "arguments": tool_call.function.arguments,  # type: ignore[union-attr]
                                        "id": tool_call.id or str(uuid.uuid4()),
                                    }
                                    for tool_call in response.choices[
                                        i
                                    ].message.tool_calls  # type: ignore[union-attr]
                                ]
                                for i in range(n)
                            ]
                        if (logprobs or top_logprobs is not None) and response.choices[
                            0
                        ].logprobs is not None:  # type: ignore[attr-defined]
                            lp = [
                                [
                                    {
                                        "token": elem.token,
                                        "logprob": elem.logprob,
                                        **(
                                            {
                                                "top_logprobs": [
                                                    {
                                                        "token": tl.token,
                                                        "logprob": tl.logprob,
                                                    }
                                                    for tl in elem.top_logprobs
                                                ]
                                            }
                                            if elem.top_logprobs
                                            else {}
                                        ),
                                    }
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
                        if (logprobs or top_logprobs is not None) and response.choices[
                            0
                        ].logprobs is not None:  # type: ignore[attr-defined]
                            lp = [
                                [
                                    {
                                        "token": token,
                                        "logprob": lp_token,
                                        **(
                                            {
                                                "top_logprobs": [
                                                    {"token": t, "logprob": lp}
                                                    for t, lp in top_lp.items()
                                                ]
                                            }
                                            if top_lp
                                            else {}
                                        ),
                                    }
                                    for token, lp_token, top_lp in zip(
                                        response.choices[i].logprobs.tokens,  # type: ignore[union-attr]
                                        response.choices[i].logprobs.token_logprobs,  # type: ignore[union-attr]
                                        [
                                            None
                                            for _ in range(
                                                len(response.choices[i].logprobs.tokens)  # type: ignore[union-attr]
                                            )
                                        ],  # top_logprobs not supported,
                                    )
                                ]
                                for i in range(n)
                            ]
                            result["response"]["logprobs"] = lp
                        if prompt_logprobs is not None and response.choices[0].prompt_logprobs is not None:  # type: ignore[attr-defined]
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
                            "request data should either have 'messages' or 'prompt'!"
                        )

                    if response.usage is not None:  # type: ignore[attr-defined]
                        result["response"]["usage"] = {
                            "prompt_tokens": response.usage.prompt_tokens,  # type: ignore[attr-defined]
                            "completion_tokens": response.usage.completion_tokens,  # type: ignore[attr-defined]
                            **(
                                {"total_tokens": response.usage.total_tokens}  # type: ignore[attr-defined]
                                if getattr(response.usage, "total_tokens", None) is not None  # type: ignore[attr-defined]
                                else {}
                            ),
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
                    import traceback

                    logger.error(traceback.format_exc())
                    exception = e
    return make_error_response(data, exception)


def batch_requests(
    url: tp.Union[str, tp.Callable[[], tp.Awaitable[str]]],
    model: str,
    requests: tp.List[tp.Dict[str, tp.Any]],
    batch_size: int | None = None,
    text_response_only: bool = False,
    verbose: bool = False,
    **kwargs,
) -> tp.List[tp.Dict[str, tp.Any] | str]:
    """
    Process multiple requests by calling make_request for each. Return the list in the same order as the requests.
    This function works whether called from sync or async context.

    Args:
        requests: list of request, each request can be different format:
          a. {"messages": [{"role": "user", "content": "hi"}]}
          b. {"prompt": "<|start_header_id|>user<|end_header_id|>\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"}
        text_response_only: if True, return the response as a list of text. otherwise return eg
        [{"request": {...}, "response": {"text": ["..."], "finish_reason": ["..."], "response_timestamp": ..., "usage": ...}}]
    """

    raw_results = run_async(
        batch_requests_async(
            func=partial(make_request, url, model, **kwargs),
            args_list=[{"data": req} for req in requests],
            batch_size=batch_size or len(requests),
            verbose=verbose,
        )
    )

    # Post-process results
    outputs: list[dict[str, tp.Any] | str] = []
    num_error = 0

    for i, result in enumerate(raw_results):
        if isinstance(result, Exception):
            result = make_error_response(requests[i], result)
        if result["response"].get("error") is not None:
            num_error += 1

        if text_response_only:
            if result["response"].get("error") is not None:
                outputs.append("")
            else:
                outputs.append(result["response"]["text"][0])
        else:
            outputs.append(result)

    logger.debug(f"complete {len(outputs)} samples, {num_error} request errors")
    return outputs


async def main(
    url: tp.Union[str, tp.Callable[[], tp.Awaitable[str]]],
    output_file: str,
    input_jsonls: str | None = None,
    app_name: str = "",
    model: str = "",
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
    batch_mode=False,
    input_hf_dataset: str | None = None,
    hf_dataset_split: str = "train",
    **kwargs,
) -> tp.Dict[str, int]:
    """Send jsonl llama3 instruct prompt for inference and save both the request and response as jsonl.
    params:
    url: Llama openai endpoint, eg http://hostname:8000/405B/v1
    output_file: name of the output jsonl file.
    input_jsonls: variable num of input jsonl files, each line is a json with two formats
        1. {text_key: prompt} if text_key is found, prompt is raw text
        2. {messages_key: Iterable[ChatCompletionMessageParam]} if messages_key is found.
    input_hf_dataset: name of a Hugging Face dataset to load directly.
    hf_dataset_split: dataset split to use when loading from Hugging Face.
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
    """

    logger.info(
        f"url: {url}, batch_size: {batch_size}, temperature: {temperature}, max_tokens: {max_tokens}, top_p: {top_p}, seed: {seed}"
    )

    save_dir = os.path.dirname(output_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(output_file):
        logger.warning(f"Output file '{output_file}' already exists, overwriting...")
    if input_hf_dataset:
        lines = load_from_hf_dataset(
            input_hf_dataset,
            hf_dataset_split,
            text_key,
            messages_key,
            system_prompt=system_prompt,
        )
    else:
        input_files = glob.glob(input_jsonls or "")
        if not input_files:
            logger.error(f"No input files found matching pattern: {input_jsonls}")
            return {}
        lines = load_from_jsonl(
            tuple(input_files),
            text_key,
            messages_key,
            system_prompt=system_prompt,
        )
    stats = {"success": 0, "total": 0, "sum_latency": 0}
    if batch_mode:
        outputs = await batch_requests_async(
            func=partial(
                make_request,
                url,
                model,
                app_name=app_name,
                seed=seed,
                top_p=top_p,
                n=n,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=logprobs,
                timeout_secs=timeout_secs,
                **kwargs,
            ),
            args_list=[{"data": line} for line in lines],
            batch_size=batch_size,
        )
        await asyncio.to_thread(
            save_to_jsonl,
            outputs,
            output_file,
            "w",
            stats,
        )
    else:
        pbar = tqdm.tqdm(total=len(lines), desc="Send request")

        pending_tasks = set()  # type: ignore
        batch_results = []
        append_output_file: bool = False

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
                    **kwargs,
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

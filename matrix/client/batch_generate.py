# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Chat-only, batch-first, multimodal-capable prompt schema similar to llm.generate in offline batch inference

Design goals:
- No vLLM dependency
- Lossless conversion to vLLM PromptType
- Supports chat + multimodal + mm_processor_kwargs
- Safe for public API exposure
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict
import typing as tp

from matrix.client.query_llm import batch_requests

# =========================
# Public Prompt Schema
# =========================


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


MultiModalValue = Any


class ChatPrompt(TypedDict, total=False):
    """
    A single chat request.
    """

    # chat models
    messages: List[ChatMessage]

    # Non-chat models
    prompt: str

    # OPTIONAL — multimodal payloads (image, video, audio, etc.)
    multi_modal_data: Dict[str, MultiModalValue]

    # OPTIONAL — multimodal processor configuration
    mm_processor_kwargs: Dict[str, Any]

    # OPTIONAL — ignored by model, useful for tracing / routing
    metadata: Dict[str, Any]


# =========================
# Public API
# =========================


def generate(
    cli,
    app_name: str,
    prompts: List[ChatPrompt],
    sampling_params: tp.Optional[tp.Dict[str, tp.Any]] = None,
    *,
    use_tqdm: bool = False,
    batch_size: int = 128,
    max_retries: int = 1000,
    text_response_only: bool = True,
) -> tp.List[tp.Dict[str, tp.Any]]:
    metadata = cli.app.get_app_metadata(app_name)

    return batch_requests(
        url=None,
        model=metadata["model_name"],
        requests=prompts,
        batch_size=batch_size,
        text_response_only=text_response_only,
        verbose=use_tqdm,
        **sampling_params,
        endpoint_cache=metadata["endpoints"]["updater"],
    )


# =========================
# Validation (Server-Side)
# =========================


def validate_messages(messages: List[ChatMessage]) -> None:
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("'messages' must be a non-empty list")

    for msg in messages:
        if msg["role"] not in ("system", "user", "assistant"):
            raise ValueError(f"Invalid role: {msg['role']}")
        if not isinstance(msg["content"], str):
            raise ValueError("Message content must be a string")


def validate_chat_prompt(prompt: ChatPrompt) -> None:
    has_messages = "messages" in prompt
    has_prompt = "prompt" in prompt

    if has_messages == has_prompt:
        raise ValueError(
            "Exactly one of 'messages' or 'prompt' must be provided"
        )

    if has_messages:
        validate_messages(prompt["messages"])

    if has_prompt:
        if not isinstance(prompt["prompt"], str):
            raise ValueError("'prompt' must be a string")

    if "multi_modal_data" in prompt:
        if not isinstance(prompt["multi_modal_data"], dict):
            raise ValueError("'multi_modal_data' must be a dict")

    if "mm_processor_kwargs" in prompt:
        if not isinstance(prompt["mm_processor_kwargs"], dict):
            raise ValueError("'mm_processor_kwargs' must be a dict")



def validate_batch(prompts: List[ChatPrompt]) -> None:
    if not isinstance(prompts, list) or len(prompts) == 0:
        raise ValueError("prompts must be a non-empty list")

    for p in prompts:
        validate_chat_prompt(p)


# =========================
# vLLM Conversion (Server-Side)
# =========================


def to_vllm_prompts(prompts: List[ChatPrompt]) -> List[Dict[str, Any]]:
    """
    Convert public ChatPrompt objects into vLLM-compatible PromptType.

    This function does NOT import vLLM.
    """
    validate_batch(prompts)

    vllm_prompts: List[Dict[str, Any]] = []

    for p in prompts:
        vp: Dict[str, Any] = {}
        if "messages" in p:
            vp["messages"] = p["messages"]

        if "prompt" in p:
            vp["prompt"] = p["prompt"]


        if "multi_modal_data" in p:
            vp["multi_modal_data"] = p["multi_modal_data"]

        if "mm_processor_kwargs" in p:
            vp["mm_processor_kwargs"] = p["mm_processor_kwargs"]

        # metadata intentionally dropped
        vllm_prompts.append(vp)

    return vllm_prompts


# =========================
# Example Usage
# =========================

if __name__ == "__main__":
    batch: List[ChatPrompt] = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Describe the image."},
            ],
            "multi_modal_data": {"image": b"...raw image bytes..."},
            "mm_processor_kwargs": {"image_size": 448},
            "metadata": {"request_id": "abc123"},
        }
    ]

    vllm_ready = to_vllm_prompts(batch)
    print(vllm_ready)

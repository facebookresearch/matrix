# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration test for Qwen3-VL batch generation with multimodal inputs.

This test:
1. Deploys Qwen3-VL model using deploy_models.py
2. Tests batch inference with text, images, and videos
3. Tests both vLLM style and OpenAI style multimodal inputs
"""

import tempfile
import time
import uuid
from typing import Any, Dict, Generator, List

import pytest

from matrix.cli import Cli
from matrix.client import batch_generate
from matrix.scripts import deploy_models
from matrix.utils.ray import status_is_pending, status_is_success


# Test URLs
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/7/75/Test_OCR_document.jpg"
VIDEO_URL = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"


@pytest.fixture(scope="module")
def qwen3vl_cluster() -> Generator[Cli, Any, Any]:
    """Start cluster and deploy Qwen3-VL model."""
    cluster_id = None # f"test_qwen3vl_{str(uuid.uuid4())[:8]}"

    # Deploy the model using deploy_models.py
    applications = [
        {
            "model_name": "/checkpoint/data/shared/pretrained-llms/Qwen3-VL-30B-A3B-Instruct",
            "use_grpc": "false",
            "min_replica": 4,
            "name": "qwen3vl",
            "model_size": "Qwen3-VL-30B-A3B-Instruct",
            "enable_tools": "true",
        }
    ]

    # Deploy using deploy_models script
    deploy_result = deploy_models.main(
        cluster_id=cluster_id,
        applications=applications,
        num_workers=1,
        slurm={"account": "data", "qos": "h100_lowest"},
        timeout=1800,
    )

    print(f"Deployment result: {deploy_result}")

    # Get the CLI instance
    cli = Cli(cluster_id=cluster_id)

    try:
        yield cli
    finally:
        # Cleanup
        try:
            pass # cli.stop_cluster()
        except Exception as e:
            print(f"Error stopping cluster: {e}")


def test_batch_generate_text(qwen3vl_cluster: Cli) -> None:
    """Test batch inference with text-only messages."""
    cli = qwen3vl_cluster
    app_name = "qwen3vl"

    # Wait for app to be ready
    for _ in range(30):
        status = cli.app.app_status(app_name)
        if status_is_success(status):
            break
        time.sleep(10)

    assert status_is_success(status), f"App not ready: {status}"

    # Create batch of text prompts
    prompts: List[batch_generate.ChatPrompt] = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
        },
        {
            "messages": [
                {"role": "user", "content": "Tell me a short joke."},
            ],
        },
    ]

    # Run batch inference
    results = batch_generate.generate(
        cli=cli,
        app_name=app_name,
        prompts=prompts,
        sampling_params={"temperature": 0.7, "max_tokens": 100},
        use_tqdm=True,
    )

    print(results)
    # Verify results
    assert len(results) == len(prompts), "Number of results doesn't match prompts"
    for result in results:
        assert "response" in result
        assert "error" not in result["response"] or result["response"]["error"] is None
        assert "text" in result["response"]
        print(f"Text response: {result['response']['text']}")


def test_batch_generate_images_vllm_style(qwen3vl_cluster: Cli) -> None:
    """Test batch inference with images using vLLM style (multi_modal_data)."""
    cli = qwen3vl_cluster
    app_name = "qwen3vl"

    # Create batch with vLLM style image inputs
    prompts: List[batch_generate.ChatPrompt] = [
        {
            "messages": [
                {"role": "user", "content": "Read all the text in the image."},
            ],
            "multi_modal_data": {"image": IMAGE_URL},
        },
        {
            "messages": [
                {"role": "user", "content": "Describe what you see in this image."},
            ],
            "multi_modal_data": {"image": IMAGE_URL},
        },
        {
            "messages": [
                {"role": "user", "content": "What items are listed in this receipt?"},
            ],
            "multi_modal_data": {"image": IMAGE_URL},
        },
    ]

    # Run batch inference
    results = batch_generate.generate(
        cli=cli,
        app_name=app_name,
        prompts=prompts,
        sampling_params={"temperature": 0.7, "max_tokens": 200},
        use_tqdm=True,
    )

    # Verify results
    assert len(results) == len(prompts), "Number of results doesn't match prompts"
    for result in results:
        assert "response" in result
        assert "error" not in result["response"] or result["response"]["error"] is None
        assert "text" in result["response"]
        print(f"Image response (vLLM style): {result['response']['text']}")


def test_batch_generate_images_openai_style(qwen3vl_cluster: Cli) -> None:
    """Test batch inference with images using OpenAI style (image_url in content)."""
    cli = qwen3vl_cluster
    app_name = "qwen3vl"

    # Create batch with OpenAI style image inputs
    # Note: OpenAI style embeds the image URL in the message content
    from matrix.client import query_llm

    metadata = cli.get_app_metadata(app_name)

    # OpenAI style requests with image_url
    prompts = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGE_URL},
                        },
                        {"type": "text", "text": "Read all the text in the image."},
                    ],
                }
            ],
            "temperature": 0.7,
            "max_tokens": 200,
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGE_URL},
                        },
                        {"type": "text", "text": "Describe what you see in this image."},
                    ],
                }
            ],
        },
    ]

    # Run batch requests using query_llm
    results = batch_generate.generate(
        cli=cli,
        app_name=app_name,
        prompts=prompts,
        sampling_params={"temperature": 0.7, "max_tokens": 200},
        use_tqdm=True,
    )
    print(results)
    # Verify results
    assert len(results) == len(prompts), "Number of results doesn't match requests"
    for result in results:
        assert "response" in result
        assert "error" not in result["response"] or result["response"]["error"] is None
        assert "text" in result["response"]
        print(f"Image response (OpenAI style): {result['response']['text']}")


def test_batch_generate_videos_vllm_style(qwen3vl_cluster: Cli) -> None:
    """Test batch inference with videos using vLLM style (multi_modal_data)."""
    cli = qwen3vl_cluster
    app_name = "qwen3vl"

    # Create batch with vLLM style video inputs
    prompts: List[batch_generate.ChatPrompt] = [
        {
            "messages": [
                {"role": "user", "content": "Describe what happens in this video."},
            ],
            "multi_modal_data": {"video": VIDEO_URL},
        },
        {
            "messages": [
                {"role": "user", "content": "What is the main action in this video?"},
            ],
            "multi_modal_data": {"video": VIDEO_URL},
        },
        {
            "messages": [
                {"role": "user", "content": "Summarize the content of this video."},
            ],
            "multi_modal_data": {"video": VIDEO_URL},
        },
    ]

    # Run batch inference
    results = batch_generate.generate(
        cli=cli,
        app_name=app_name,
        prompts=prompts,
        sampling_params={"temperature": 0.7, "max_tokens": 200},
        use_tqdm=True,
    )

    # Verify results
    assert len(results) == len(prompts), "Number of results doesn't match prompts"
    for result in results:
        assert "response" in result
        assert "error" not in result["response"] or result["response"]["error"] is None
        assert "text" in result["response"]
        print(f"Video response (vLLM style): {result['response']['text']}")


def test_batch_generate_videos_openai_style(qwen3vl_cluster: Cli) -> None:
    """Test batch inference with videos using OpenAI style."""
    cli = qwen3vl_cluster
    app_name = "qwen3vl"

    from matrix.client import query_llm

    metadata = cli.get_app_metadata(app_name)

    # OpenAI style requests with video URL
    # Note: Using image_url type but with video URL (vLLM supports this)
    openai_requests = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": VIDEO_URL},
                        },
                        {"type": "text", "text": "Describe what happens in this video."},
                    ],
                }
            ],
            "temperature": 0.7,
            "max_tokens": 200,
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": VIDEO_URL},
                        },
                        {"type": "text", "text": "What is the main action in this video?"},
                    ],
                }
            ],
        },
    ]

    # Run batch requests using query_llm
    results = batch_generate.generate(
        cli=cli,
        app_name=app_name,
        prompts=prompts,
        sampling_params={"temperature": 0.7, "max_tokens": 200},
        use_tqdm=True,
    )

    print(results)
    # Verify results
    assert len(results) == len(prompts), "Number of results doesn't match requests"
    for result in results:
        assert "response" in result
        assert "error" not in result["response"] or result["response"]["error"] is None
        assert "text" in result["response"]
        print(f"Video response (OpenAI style): {result['response']['text']}")


def test_batch_generate_mixed_modalities(qwen3vl_cluster: Cli) -> None:
    """Test batch inference with mixed modalities (text, images, videos)."""
    cli = qwen3vl_cluster
    app_name = "qwen3vl"

    # Create batch with mixed inputs
    prompts: List[batch_generate.ChatPrompt] = [
        # Text only
        {
            "messages": [
                {"role": "user", "content": "What is 5+3?"},
            ],
        },
        # Image
        {
            "messages": [
                {"role": "user", "content": "Read all the text in the image."},
            ],
            "multi_modal_data": {"image": IMAGE_URL},
        },
        # Video
        {
            "messages": [
                {"role": "user", "content": "Describe what happens in this video."},
            ],
            "multi_modal_data": {"video": VIDEO_URL},
        },
        # Another text
        {
            "messages": [
                {"role": "user", "content": "Tell me a fun fact."},
            ],
        },
        # Another image with different prompt
        {
            "messages": [
                {"role": "user", "content": "What type of document is this?"},
            ],
            "multi_modal_data": {"image": IMAGE_URL},
        },
    ]

    # Run batch inference
    results = batch_generate.generate(
        cli=cli,
        app_name=app_name,
        prompts=prompts,
        sampling_params={"temperature": 0.7, "max_tokens": 200},
        use_tqdm=True,
        batch_size=5,
    )

    # Verify results
    assert len(results) == len(prompts), "Number of results doesn't match prompts"
    for i, result in enumerate(results):
        assert "response" in result
        assert "error" not in result["response"] or result["response"]["error"] is None
        assert "text" in result["response"]
        print(f"Mixed batch result {i}: {result['response']['text']}")

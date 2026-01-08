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

import base64
import tempfile
import time
import uuid
from typing import Any, Dict, Generator, List

import pytest

from matrix.cli import Cli
from matrix.client import batch_generate
from matrix.scripts import deploy_models
from matrix.utils.ray import status_is_pending, status_is_success


# Test media files - local paths
IMAGE_PATH = "/checkpoint/data/shared/matrix_cluster/receipt.png"
VIDEO_PATH = "/checkpoint/data/shared/matrix_cluster/ForBiggerBlazes.mp4"


def encode_file_to_base64(file_path: str, media_type: str = "image") -> str:
    """
    Load media from local file and encode as base64 data URL.

    Args:
        file_path: Local file path
        media_type: Either "image" or "video"

    Returns:
        Base64 data URL string (e.g., "data:image/jpeg;base64,...")
    """
    print(f"Loading {media_type} from local file {file_path}...")
    with open(file_path, "rb") as f:
        media_data = f.read()

    # Encode to base64
    base64_data = base64.b64encode(media_data).decode('utf-8')

    # Determine MIME type from file extension
    if media_type == "image":
        if file_path.endswith('.png'):
            mime_type = "image/png"
        elif file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
            mime_type = "image/jpeg"
        elif file_path.endswith('.gif'):
            mime_type = "image/gif"
        else:
            mime_type = "image/jpeg"  # default
    else:  # video
        if file_path.endswith('.mp4'):
            mime_type = "video/mp4"
        elif file_path.endswith('.webm'):
            mime_type = "video/webm"
        else:
            mime_type = "video/mp4"  # default

    data_url = f"data:{mime_type};base64,{base64_data}"
    print(f"Encoded {media_type} to base64 ({len(base64_data)} chars)")
    return data_url


# Load and cache the media as base64 data URLs
# These will be loaded once when the fixture runs
IMAGE_BASE64 = None
VIDEO_BASE64 = None


@pytest.fixture(scope="module")
def qwen3vl_cluster() -> Generator[Cli, Any, Any]:
    """Start cluster and deploy Qwen3-VL model."""
    global IMAGE_BASE64, VIDEO_BASE64

    # Load media files from local disk and convert to base64
    # This happens on the client machine, then base64 is sent to vLLM server
    try:
        IMAGE_BASE64 = encode_file_to_base64(IMAGE_PATH, media_type="image")
    except Exception as e:
        print(f"Warning: Could not load image: {e}")
        IMAGE_BASE64 = None

    try:
        VIDEO_BASE64 = encode_file_to_base64(VIDEO_PATH, media_type="video")
    except Exception as e:
        print(f"Warning: Could not load video: {e}")
        VIDEO_BASE64 = None

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
        text_response_only=False,
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

    if IMAGE_BASE64 is None:
        pytest.skip("Image could not be loaded")

    # Create batch with vLLM style image inputs using base64 data URL
    prompts: List[batch_generate.ChatPrompt] = [
        {
            "messages": [
                {"role": "user", "content": "Read all the text in the image."},
            ],
            "multi_modal_data": {"image": IMAGE_BASE64},
        },
        {
            "messages": [
                {"role": "user", "content": "Describe what you see in this image."},
            ],
            "multi_modal_data": {"image": IMAGE_BASE64},
        },
        {
            "messages": [
                {"role": "user", "content": "What items are listed in this receipt?"},
            ],
            "multi_modal_data": {"image": IMAGE_BASE64},
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
        assert result
        print(f"Image response (vLLM style): {result}")


def test_batch_generate_images_openai_style(qwen3vl_cluster: Cli) -> None:
    """Test batch inference with images using OpenAI style (image_url in content)."""
    cli = qwen3vl_cluster
    app_name = "qwen3vl"

    if IMAGE_BASE64 is None:
        pytest.skip("Image could not be loaded")

    # Create batch with OpenAI style image inputs
    # Note: OpenAI style embeds the image in the message content
    # Base64 data URLs work here too!
    prompts = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGE_BASE64},  # Use base64 data URL
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
                            "image_url": {"url": IMAGE_BASE64},  # Use base64 data URL
                        },
                        {"type": "text", "text": "Describe what you see in this image."},
                    ],
                }
            ],
        },
    ]

    # Run batch requests using batch_generate
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
        assert result
        print(f"Image response (OpenAI style): {result}")


def test_batch_generate_videos_vllm_style(qwen3vl_cluster: Cli) -> None:
    """Test batch inference with videos using vLLM style (multi_modal_data)."""
    cli = qwen3vl_cluster
    app_name = "qwen3vl"

    if VIDEO_BASE64 is None:
        pytest.skip("Video could not be loaded")

    # Create batch with vLLM style video inputs using base64 data URL
    prompts: List[batch_generate.ChatPrompt] = [
        {
            "messages": [
                {"role": "user", "content": "Describe what happens in this video."},
            ],
            "multi_modal_data": {"video": VIDEO_BASE64},
        },
        {
            "messages": [
                {"role": "user", "content": "What is the main action in this video?"},
            ],
            "multi_modal_data": {"video": VIDEO_BASE64},
        },
        {
            "messages": [
                {"role": "user", "content": "Summarize the content of this video."},
            ],
            "multi_modal_data": {"video": VIDEO_BASE64},
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
        assert result
        print(f"Video response (vLLM style): {result}")


def test_batch_generate_videos_openai_style(qwen3vl_cluster: Cli) -> None:
    """Test batch inference with videos using OpenAI style."""
    cli = qwen3vl_cluster
    app_name = "qwen3vl"

    if VIDEO_BASE64 is None:
        pytest.skip("Video could not be loaded")

    # OpenAI style requests with video base64 data URL
    # Try using type: "video_url" for videos (vLLM extension)
    prompts = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",  # Use video_url for videos
                            "video_url": {"url": VIDEO_BASE64},
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
                            "type": "video_url",  # Use video_url for videos
                            "video_url": {"url": VIDEO_BASE64},
                        },
                        {"type": "text", "text": "What is the main action in this video?"},
                    ],
                }
            ],
        },
    ]

    # Run batch requests using batch_generate
    results = batch_generate.generate(
        cli=cli,
        app_name=app_name,
        prompts=prompts,
        sampling_params={"temperature": 0.7, "max_tokens": 200},
        use_tqdm=True,
        text_response_only=True,
    )

    print(results)
    # Verify results
    assert len(results) == len(prompts), "Number of results doesn't match requests"
    for result in results:
        assert result
        print(f"Video response (OpenAI style): {result}")


def test_batch_generate_mixed_modalities(qwen3vl_cluster: Cli) -> None:
    """Test batch inference with mixed modalities (text, images, videos)."""
    cli = qwen3vl_cluster
    app_name = "qwen3vl"

    if IMAGE_BASE64 is None or VIDEO_BASE64 is None:
        pytest.skip("Image or video could not be loaded")

    # Create batch with mixed inputs using base64 data URLs
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
            "multi_modal_data": {"image": IMAGE_BASE64},
        },
        # Video
        {
            "messages": [
                {"role": "user", "content": "Describe what happens in this video."},
            ],
            "multi_modal_data": {"video": VIDEO_BASE64},
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
            "multi_modal_data": {"image": IMAGE_BASE64},
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
        assert result
        print(f"Mixed batch result {i}: {result}")

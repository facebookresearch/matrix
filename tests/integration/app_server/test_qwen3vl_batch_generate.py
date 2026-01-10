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
from matrix.client import query_llm
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
    base64_data = base64.b64encode(media_data).decode("utf-8")

    # Determine MIME type from file extension
    if media_type == "image":
        if file_path.endswith(".png"):
            mime_type = "image/png"
        elif file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
            mime_type = "image/jpeg"
        elif file_path.endswith(".gif"):
            mime_type = "image/gif"
        else:
            mime_type = "image/jpeg"  # default
    else:  # video
        if file_path.endswith(".mp4"):
            mime_type = "video/mp4"
        elif file_path.endswith(".webm"):
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

    cluster_id = f"test_qwen3vl_{str(uuid.uuid4())[:8]}"

    # Deploy the model using deploy_models.py
    applications = [
        {
            "model_name": "/checkpoint/data/shared/pretrained-llms/Qwen3-VL-30B-A3B-Instruct",
            "use_grpc": "false",
            "min_replica": 4,
            "name": "qwen3vl",
            "model_size": "Qwen3-VL-30B-A3B-Instruct",
            "enable_tools": "true",
            "allowed-local-media-path": "/",
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
            cli.stop_cluster()
        except Exception as e:
            print(f"Error stopping cluster: {e}")


def test_batch_generate_text(qwen3vl_cluster: Cli) -> None:
    """Test batch inference with text-only messages."""
    cli = qwen3vl_cluster
    app_name = "qwen3vl"

    # Wait for app to be ready
    for _ in range(30):
        status, _ = cli.app.app_status(app_name)
        if status_is_success(status):
            break
        time.sleep(10)

    assert status_is_success(status), f"App not ready: {status}"

    # Create batch of text prompts
    prompts: List[Dict[str, Any]] = [
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

    # Run batch inference (text_response_only=False returns full response objects)
    results = query_llm.generate(
        cli=cli,
        app_name=app_name,
        prompts=prompts,
        sampling_params={"temperature": 0.7, "max_tokens": 100},
        use_tqdm=True,
        text_response_only=False,
    )

    # Verify results
    assert len(results) == len(prompts), "Number of results doesn't match prompts"
    for i, result in enumerate(results):
        assert "response" in result
        assert "error" not in result["response"] or result["response"]["error"] is None
        assert "text" in result["response"]
        # When text_response_only=False, response.text is a string
        text = result["response"]["text"][0]
        assert isinstance(text, str), f"Expected str, got {type(text)}"
        assert text and len(text.strip()) > 0, f"Empty response for prompt {i}"
        print(f"Text response {i}: {text}")


def test_batch_generate_images_openai_style(qwen3vl_cluster: Cli) -> None:
    """Test batch inference with images using OpenAI style (image_url in content)."""
    cli = qwen3vl_cluster
    app_name = "qwen3vl"

    if IMAGE_BASE64 is None:
        pytest.skip("Image could not be loaded")

    # Create batch with OpenAI style image inputs
    # Note: OpenAI style embeds the image in the message content
    # Test both base64 data URLs and file paths
    prompts = [
        # Using base64 data URL
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
        },
        # Using file path
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"file://{IMAGE_PATH}"
                            },  # Use file:// URL
                        },
                        {
                            "type": "text",
                            "text": "Describe what you see in this image.",
                        },
                    ],
                }
            ],
        },
    ]

    # Run batch requests (text_response_only=True by default, returns list of strings)
    results = query_llm.generate(
        cli=cli,
        app_name=app_name,
        prompts=prompts,
        sampling_params={"temperature": 0.7, "max_tokens": 512},
        use_tqdm=True,
    )

    # Verify results
    assert len(results) == len(prompts), "Number of results doesn't match requests"

    # Expected keywords in receipt image (case-insensitive)
    receipt_keywords = ["subtotal", "sub total", "change", "due", "total"]

    for i, result in enumerate(results):
        # When text_response_only=True (default), result is a string
        assert isinstance(
            result, str
        ), f"Result {i} should be a string when text_response_only=True"
        assert result and len(result.strip()) > 0, f"Empty text response for prompt {i}"

        # Check for receipt-related content (case-insensitive)
        result_lower = result.lower()
        found_keywords = [kw for kw in receipt_keywords if kw in result_lower]
        assert (
            len(found_keywords) > 0
        ), f"Response {i} missing receipt keywords. Expected any of {receipt_keywords}, got: {result[:200]}"

        print(f"Image response {i} (found keywords: {found_keywords}): {result}")


def test_batch_generate_videos_openai_style(qwen3vl_cluster: Cli) -> None:
    """Test batch inference with videos using OpenAI style."""
    cli = qwen3vl_cluster
    app_name = "qwen3vl"

    if VIDEO_BASE64 is None:
        pytest.skip("Video could not be loaded")

    # OpenAI style requests with videos
    # Note: Use "image_url" type even for videos - vLLM detects video from MIME type
    # Test both base64 data URLs and file paths
    prompts = [
        # Using base64 data URL
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": VIDEO_BASE64
                            },  # vLLM detects video from data:video/mp4
                        },
                        {
                            "type": "text",
                            "text": "Describe what happens in this video.",
                        },
                    ],
                }
            ],
        },
        # Using file path
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"file://{VIDEO_PATH}"
                            },  # Use file:// URL
                        },
                        {
                            "type": "text",
                            "text": "What is the main action in this video?",
                        },
                    ],
                }
            ],
        },
    ]

    # Run batch requests (text_response_only=True returns strings)
    results = query_llm.generate(
        cli=cli,
        app_name=app_name,
        prompts=prompts,
        sampling_params={"temperature": 0.7, "max_tokens": 512},
        use_tqdm=True,
        text_response_only=True,
    )

    # Verify results
    assert len(results) == len(prompts), "Number of results doesn't match requests"

    # Expected keywords in video (case-insensitive)
    video_keywords = ["phone", "cell phone", "tablet", "dragon", "chromecast", "tv"]

    for i, result in enumerate(results):
        # When text_response_only=True, result is a string
        assert isinstance(
            result, str
        ), f"Result {i} should be a string when text_response_only=True"
        assert result and len(result.strip()) > 0, f"Empty text response for prompt {i}"

        # Check for video-related content (case-insensitive)
        result_lower = result.lower()
        found_keywords = [kw for kw in video_keywords if kw in result_lower]
        assert (
            len(found_keywords) > 0
        ), f"Response {i} missing video keywords. Expected any of {video_keywords}, got: {result[:200]}"

        print(f"Video response {i} (found keywords: {found_keywords}): {result}")

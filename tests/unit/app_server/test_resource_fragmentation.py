# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for resource fragmentation fix (Issue #111).

This test verifies that applications are sorted by GPU requirements
to minimize resource fragmentation when deploying multiple models
with different GPU requirements.
"""

import pytest

from matrix.app_server.deploy_utils import (
    get_gpu_requirements_per_replica,
    sort_apps_by_gpu_requirements,
)


def test_get_gpu_requirements_per_replica():
    """Test GPU requirement calculation for different app types."""
    # LLM model with explicit tensor_parallel_size
    app1 = {
        "app_type": "llm",
        "model_name": "test-model",
        "tensor-parallel-size": 4,
    }
    assert get_gpu_requirements_per_replica(app1) == 4

    # LLM model with tensor_parallel_size (underscore format)
    app2 = {
        "app_type": "llm",
        "model_name": "test-model",
        "tensor_parallel_size": 2,
    }
    assert get_gpu_requirements_per_replica(app2) == 2

    # LLM model using default from llm_config
    app3 = {
        "app_type": "llm",
        "model_name": "meta-llama/Llama-3.1-70B-Instruct",
    }
    # Llama-3.1-70B-Instruct has tensor-parallel-size: 4 in defaults
    assert get_gpu_requirements_per_replica(app3) == 4

    # Vision model
    app4 = {"app_type": "perception_encoder", "model_name": "test-vision"}
    assert get_gpu_requirements_per_replica(app4) == 1

    # Code execution (no GPU)
    app5 = {"app_type": "code", "name": "code"}
    assert get_gpu_requirements_per_replica(app5) == 0

    # Container (no GPU)
    app6 = {"app_type": "container", "name": "container"}
    assert get_gpu_requirements_per_replica(app6) == 0


def test_sort_apps_by_gpu_requirements():
    """Test that apps are sorted by GPU requirements (largest first)."""
    # Create apps with different GPU requirements
    apps = [
        {
            "app_type": "llm",
            "model_name": "model-1gpu",
            "tensor-parallel-size": 1,
            "name": "model-a",
            "min_replica": 14,
        },
        {
            "app_type": "llm",
            "model_name": "model-2gpu",
            "tensor-parallel-size": 2,
            "name": "model-b",
            "min_replica": 1,
        },
        {
            "app_type": "llm",
            "model_name": "model-4gpu",
            "tensor-parallel-size": 4,
            "name": "model-c",
            "min_replica": 1,
        },
        {
            "app_type": "code",
            "name": "code-app",
        },
    ]

    sorted_apps = sort_apps_by_gpu_requirements(apps)

    # Verify sorting: largest GPU requirement first
    assert sorted_apps[0]["name"] == "model-c"  # 4 GPUs
    assert sorted_apps[1]["name"] == "model-b"  # 2 GPUs
    assert sorted_apps[2]["name"] == "model-a"  # 1 GPU
    assert sorted_apps[3]["name"] == "code-app"  # 0 GPUs

    # Verify all apps are present
    assert len(sorted_apps) == len(apps)


def test_sort_apps_same_gpu_requirements():
    """Test sorting when apps have same GPU requirements."""
    apps = [
        {
            "app_type": "llm",
            "model_name": "model-a",
            "tensor-parallel-size": 2,
            "name": "model-a",
            "min_replica": 1,
        },
        {
            "app_type": "llm",
            "model_name": "model-b",
            "tensor-parallel-size": 2,
            "name": "model-b",
            "min_replica": 5,
        },
    ]

    sorted_apps = sort_apps_by_gpu_requirements(apps)

    # When GPU requirements are equal, sort by min_replica (descending)
    assert sorted_apps[0]["name"] == "model-b"  # min_replica: 5
    assert sorted_apps[1]["name"] == "model-a"  # min_replica: 1


def test_sort_apps_with_defaults():
    """Test sorting when apps use default tensor_parallel_size from model config."""
    apps = [
        {
            "app_type": "llm",
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",  # Default: 1 GPU
            "name": "model-8b",
        },
        {
            "app_type": "llm",
            "model_name": "meta-llama/Llama-3.1-70B-Instruct",  # Default: 4 GPUs
            "name": "model-70b",
        },
    ]

    sorted_apps = sort_apps_by_gpu_requirements(apps)

    # 70B model (4 GPUs) should come before 8B model (1 GPU)
    assert sorted_apps[0]["name"] == "model-70b"
    assert sorted_apps[1]["name"] == "model-8b"


def test_sort_single_app():
    """Test that sorting a single app returns it unchanged."""
    apps = [
        {
            "app_type": "llm",
            "model_name": "test-model",
            "tensor-parallel-size": 2,
            "name": "single-app",
        }
    ]

    sorted_apps = sort_apps_by_gpu_requirements(apps)

    assert len(sorted_apps) == 1
    assert sorted_apps[0]["name"] == "single-app"


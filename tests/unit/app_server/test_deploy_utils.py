# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from matrix.app_server.deploy_utils import get_resource_requirements


class TestGetResourceRequirements:
    """Unit tests for get_resource_requirements function."""

    # LLM app tests
    def test_llm_default_tp_pp(self):
        """LLM app with default tp=1, pp=1."""
        app_config = {
            "app_type": "llm",
            "model_name": "unknown-model",
            "min_replica": 2,
        }
        result = get_resource_requirements(app_config)
        # GPU: 1 * 1 * 2 = 2
        # CPU: 2 * (1 + 1 * 4) = 10
        assert result["GPU"] == 2.0
        assert result["CPU"] == 10.0

    def test_llm_explicit_tp_pp(self):
        """LLM app with explicit tp and pp values."""
        app_config = {
            "app_type": "llm",
            "model_name": "some-model",
            "tensor-parallel-size": 4,
            "pipeline-parallel-size": 2,
            "min_replica": 1,
        }
        result = get_resource_requirements(app_config)
        # GPU: 4 * 2 * 1 = 8
        # CPU: 1 * (1 + 8 * 4) = 33
        assert result["GPU"] == 8.0
        assert result["CPU"] == 33.0

    def test_llm_with_ray_resources_num_cpus(self):
        """LLM app with ray_resources overriding cpus_per_gpu."""
        app_config = {
            "app_type": "llm",
            "model_name": "some-model",
            "tensor-parallel-size": 2,
            "pipeline-parallel-size": 1,
            "min_replica": 4,
            "ray_resources": {"num_cpus": 2},
        }
        result = get_resource_requirements(app_config)
        # GPU: 2 * 1 * 4 = 8
        # CPU: 4 * (1 + 2 * 2) = 20 (instead of 4 * (1 + 2 * 4) = 36)
        assert result["GPU"] == 8.0
        assert result["CPU"] == 20.0

    def test_llm_with_ray_resources_as_json_string(self):
        """LLM app with ray_resources as JSON string."""
        app_config = {
            "app_type": "llm",
            "model_name": "some-model",
            "tensor-parallel-size": 2,
            "pipeline-parallel-size": 1,
            "min_replica": 2,
            "ray_resources": '{"num_cpus": 1}',
        }
        result = get_resource_requirements(app_config)
        # GPU: 2 * 1 * 2 = 4
        # CPU: 2 * (1 + 2 * 1) = 6
        assert result["GPU"] == 4.0
        assert result["CPU"] == 6.0

    def test_llm_lookup_by_model_size(self):
        """LLM app looking up tp/pp by model_size matching 'name' field."""
        app_config = {
            "app_type": "llm",
            "model_name": "/some/custom/path/model",
            "model_size": "70B",  # matches name "70B" in llm_model_default_parameters
            "min_replica": 1,
        }
        result = get_resource_requirements(app_config)
        # 70B model has tp=4, pp=1
        # GPU: 4 * 1 * 1 = 4
        # CPU: 1 * (1 + 4 * 4) = 17
        assert result["GPU"] == 4.0
        assert result["CPU"] == 17.0

    def test_llm_lookup_by_exact_model_name(self):
        """LLM app looking up tp/pp by exact model_name match."""
        app_config = {
            "app_type": "llm",
            "model_name": "meta-llama/Llama-3.1-70B-Instruct",
            "min_replica": 2,
        }
        result = get_resource_requirements(app_config)
        # 70B model has tp=4, pp=1
        # GPU: 4 * 1 * 2 = 8
        # CPU: 2 * (1 + 4 * 4) = 34
        assert result["GPU"] == 8.0
        assert result["CPU"] == 34.0

    def test_sglang_llm(self):
        """sglang_llm app type."""
        app_config = {
            "app_type": "sglang_llm",
            "model_name": "some-model",
            "tensor-parallel-size": 2,
            "pipeline-parallel-size": 1,
            "min_replica": 3,
        }
        result = get_resource_requirements(app_config)
        # GPU: 2 * 1 * 3 = 6
        # CPU: 3 * (1 + 2 * 4) = 27
        assert result["GPU"] == 6.0
        assert result["CPU"] == 27.0

    def test_fastgen(self):
        """fastgen app type."""
        app_config = {
            "app_type": "fastgen",
            "model_name": "some-model",
            "tensor-parallel-size": 1,
            "pipeline-parallel-size": 1,
            "min_replica": 4,
        }
        result = get_resource_requirements(app_config)
        # GPU: 1 * 1 * 4 = 4
        # CPU: 4 * (1 + 1 * 4) = 20
        assert result["GPU"] == 4.0
        assert result["CPU"] == 20.0

    # Vision app tests
    def test_perception_encoder(self):
        """perception_encoder app - 1 GPU per replica."""
        app_config = {
            "app_type": "perception_encoder",
            "model_name": "some-encoder",
            "min_replica": 4,
        }
        result = get_resource_requirements(app_config)
        assert result["GPU"] == 4.0
        assert result["CPU"] == 4.0

    def test_optical_flow(self):
        """optical_flow app - 1 GPU per replica."""
        app_config = {
            "app_type": "optical_flow",
            "model_name": "some-flow-model",
            "min_replica": 2,
        }
        result = get_resource_requirements(app_config)
        assert result["GPU"] == 2.0
        assert result["CPU"] == 2.0

    # Container app tests
    def test_container_with_gpu(self):
        """Container app with GPU resources."""
        app_config = {
            "app_type": "container",
            "name": "my-container",
            "min_replica": 2,
            "ray_resources": {"GPU": 2, "CPU": 8},
        }
        result = get_resource_requirements(app_config)
        assert result["GPU"] == 4.0  # 2 * 2
        assert result["CPU"] == 16.0  # 8 * 2

    def test_container_with_ray_resources_json_string(self):
        """Container app with ray_resources as JSON string."""
        app_config = {
            "app_type": "container",
            "name": "my-container",
            "min_replica": 3,
            "ray_resources": '{"GPU": 1, "CPU": 4}',
        }
        result = get_resource_requirements(app_config)
        assert result["GPU"] == 3.0  # 1 * 3
        assert result["CPU"] == 12.0  # 4 * 3

    def test_container_cpu_only(self):
        """Container app with CPU only."""
        app_config = {
            "app_type": "container",
            "name": "my-container",
            "min_replica": 4,
            "ray_resources": {"CPU": 2},
        }
        result = get_resource_requirements(app_config)
        assert result["GPU"] == 0.0
        assert result["CPU"] == 8.0

    # CPU-only app tests
    def test_code_app(self):
        """Code execution app - CPU only."""
        app_config = {
            "app_type": "code",
            "name": "code",
            "min_replica": 8,
        }
        result = get_resource_requirements(app_config)
        assert result["GPU"] == 0.0
        assert result["CPU"] == 8.0

    def test_hello_app(self):
        """Hello app - CPU only."""
        app_config = {
            "app_type": "hello",
            "name": "hello",
            "min_replica": 1,
        }
        result = get_resource_requirements(app_config)
        assert result["GPU"] == 0.0
        assert result["CPU"] == 1.0

    def test_openai_proxy(self):
        """OpenAI proxy app - CPU only."""
        app_config = {
            "app_type": "openai",
            "name": "openai",
            "min_replica": 2,
        }
        result = get_resource_requirements(app_config)
        assert result["GPU"] == 0.0
        assert result["CPU"] == 2.0

    def test_gemini_proxy(self):
        """Gemini proxy app - CPU only."""
        app_config = {
            "app_type": "gemini",
            "name": "gemini",
            "min_replica": 3,
        }
        result = get_resource_requirements(app_config)
        assert result["GPU"] == 0.0
        assert result["CPU"] == 3.0

    def test_bedrock_proxy(self):
        """Bedrock proxy app - CPU only."""
        app_config = {
            "app_type": "bedrock",
            "name": "bedrock",
            "min_replica": 2,
        }
        result = get_resource_requirements(app_config)
        assert result["GPU"] == 0.0
        assert result["CPU"] == 2.0

    # Default behavior tests
    def test_default_app_type_is_llm(self):
        """App without app_type defaults to llm."""
        app_config = {
            "model_name": "some-model",
            "tensor-parallel-size": 1,
            "pipeline-parallel-size": 1,
            "min_replica": 2,
        }
        result = get_resource_requirements(app_config)
        # Should be treated as LLM
        assert result["GPU"] == 2.0
        assert result["CPU"] == 10.0  # 2 * (1 + 1 * 4)

    def test_default_min_replica_is_1(self):
        """App without min_replica defaults to 1."""
        app_config = {
            "app_type": "code",
            "name": "code",
        }
        result = get_resource_requirements(app_config)
        assert result["GPU"] == 0.0
        assert result["CPU"] == 1.0

    # Edge cases
    def test_invalid_ray_resources_json(self):
        """Invalid JSON string for ray_resources falls back to defaults."""
        app_config = {
            "app_type": "llm",
            "model_name": "some-model",
            "tensor-parallel-size": 2,
            "pipeline-parallel-size": 1,
            "min_replica": 1,
            "ray_resources": "invalid json",
        }
        result = get_resource_requirements(app_config)
        # Falls back to default CPUS_PER_GPU = 4
        assert result["GPU"] == 2.0
        assert result["CPU"] == 9.0  # 1 * (1 + 2 * 4)

    def test_large_scale_deployment(self):
        """Large scale deployment with 405B model."""
        app_config = {
            "app_type": "llm",
            "model_name": "meta-llama/Llama-3.1-405B-Instruct",
            "min_replica": 1,
        }
        result = get_resource_requirements(app_config)
        # 405B model has tp=8, pp=2
        # GPU: 8 * 2 * 1 = 16
        # CPU: 1 * (1 + 16 * 4) = 65
        assert result["GPU"] == 16.0
        assert result["CPU"] == 65.0

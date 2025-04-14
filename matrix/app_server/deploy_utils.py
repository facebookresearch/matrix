# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import copy
import json
import os
import signal
import subprocess
import threading
from typing import Any, Awaitable, Dict, List, Optional, Union

import aiohttp
import ray
import yaml
from jinja2 import Template
from ray import serve

from matrix.app_server.llm.ray_serve_vllm import BaseDeployment
from matrix.common.cluster_info import ClusterInfo
from matrix.utils.ray import Action, get_ray_address, kill_matrix_actors

common_config = """
proxy_location: EveryNode
http_options:
  host: 0.0.0.0
  port: {{ http_port }}
  request_timeout_s: 3600

grpc_options:
  port: {{ grpc_port }}
  grpc_servicer_functions:
    - matrix.app_server.llm.openai_pb2_grpc.add_OpenaiServiceServicer_to_server

logging_config:
  encoding: TEXT
  log_level: INFO
  logs_dir: null
  enable_access_log: true

applications:
"""

# default model parameters can be overwritten from command line
llama_model_default_parameters = {
    "meta-llama/Meta-Llama-3.1-3B-Instruct": {
        "name": "3B",
        "tensor-parallel-size": 1,
        "pipeline-parallel-size": 1,
        "enable-prefix-caching": True,
        "max_ongoing_requests": 256,
        "max-model-len": 131072,
        "gpu-memory-utilization": 0.8,
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "name": "8B",
        "tensor-parallel-size": 1,
        "pipeline-parallel-size": 1,
        "enable-prefix-caching": True,
        "max_ongoing_requests": 150,
        "max-model-len": 131072,
        "gpu-memory-utilization": 0.8,
    },
    "meta-llama/Meta-Llama-3.1-70B-Instruct": {
        "name": "70B",
        "tensor-parallel-size": 4,
        "pipeline-parallel-size": 1,
        "enable-prefix-caching": True,
        "max-model-len": 30960,
        "gpu-memory-utilization": 0.8,
        "max_ongoing_requests": 100,
    },
    "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8": {
        "name": "405B-FP8",
        "tensor-parallel-size": 8,
        "pipeline-parallel-size": 1,
        "enable-prefix-caching": True,
        "max-model-len": 10240,
        "gpu-memory-utilization": 0.8,
        "max_ongoing_requests": 50,
    },
    "meta-llama/Meta-Llama-3.1-405B-Instruct": {
        "name": "405B",
        "tensor-parallel-size": 8,
        "pipeline-parallel-size": 2,
        "enable-prefix-caching": True,
        "max-model-len": 10240,  # 30960 (4 node), 61440 (6 node), 128000 (10 nodes)
        "gpu-memory-utilization": 0.8,
        "max_ongoing_requests": 50,
    },
    "meta-llama/Meta-Llama-3.3-70B-Instruct": {
        "name": "3_3_70B",
        "tensor-parallel-size": 4,
        "pipeline-parallel-size": 1,
        "enable-prefix-caching": True,
        "max-model-len": 30960,
        "gpu-memory-utilization": 0.8,
        "max_ongoing_requests": 100,
    },
    "deepseek-ai/DeepSeek-R1": {
        "name": "deepseek-r1",
        "tensor-parallel-size": 8,
        "pipeline-parallel-size": 3,
        "enable-prefix-caching": True,
        "max-model-len": 32768,
        "gpu-memory-utilization": 0.9,
        "max_ongoing_requests": 80,
        "trust-remote-code": True,
    },
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": {
        "name": "scout",
        "tensor-parallel-size": 4,
        "pipeline-parallel-size": 1,
        "enable-prefix-caching": True,
        "max-model-len": 32768,
        "gpu-memory-utilization": 0.8,
        "max_ongoing_requests": 100,
        "use_v1_engine": "true",
    },
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": {
        "name": "maverick-fp8",
        "tensor-parallel-size": 8,
        "pipeline-parallel-size": 1,
        "enable-prefix-caching": True,
        "max-model-len": 128000,
        "gpu-memory-utilization": 0.8,
        "max_ongoing_requests": 100,
        "dtype": "auto",
        "kv-cache-dtype": "auto",
        "quantization": "compressed-tensors",
        "use_v1_engine": "true",
    },
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit": {
        "name": "unsloth-mistral-7B",
        "tensor-parallel-size": 1,
        "pipeline-parallel-size": 1,
        "enable-prefix-caching": True,
        "max_ongoing_requests": 256,
        "max-model-len": 32768,
        "gpu-memory-utilization": 0.4,
        "enable-lora": True,
        "quantization": "bitsandbytes",
        "load-format": "bitsandbytes",
        "max_lora_rank": 32,
    },
}

non_model_params = [
    "model_name",
    "name",
    "app_type",
    "min_replica",
    "max_replica",
    "pythonpath",
    "model_size",
    "max_ongoing_requests",
    "api_version",
    "api_endpoint",
    "api_key",
    "use_grpc",
    "access_token",
    "aws_account",
    "aws_region",
    "endpoint_name",
]

vllm_app_template = """
- name: {{ app.name }}
  route_prefix: /{{ app.name }}
  import_path: matrix.app_server.llm.ray_serve_vllm:{{ 'build_app_grpc' if app.use_grpc else 'build_app' }}
  runtime_env:
    env_vars:
        OUTLINES_CACHE_DIR: {{ temp_dir }}/.outlines
        RAY_DEBUG: legacy
  args:
    model: {{ app.model_name }}
    {% for key, value in app.items() %}
    {% if key not in non_model_params %}
    {{ key }}: {{ 'null' if value is true else value }}
    {% endif %}
    {% endfor %}
  deployments:
  {% if app.use_grpc %}
  - name: GrpcDeployment
  {% elif app.app_type == 'sglang_llm' %}
  - name: SglangDeployment
  {% else %}
  - name: VLLMDeployment
  {% endif %}
    max_ongoing_requests: {{ app.max_ongoing_requests }}
    autoscaling_config:
      target_ongoing_requests: {{ (app.max_ongoing_requests * 0.8) | int }}
      min_replicas: {{ app.min_replica }}
      max_replicas: {{ app.max_replica }}
"""
other_app_template = """
{% if app.app_type == 'openai' %}
- name: {{ app.name }}
  route_prefix: /{{ app.name }}
  import_path: matrix.app_server.llm.azure_openai_proxy:build_app
  args:
    model: {{ app.model_name }}
    api_version: "{{ app.api_version }}"
    api_endpoint: {{ app.api_endpoint }}
    api_key: {{ app.api_key }}
  deployments:
  - name: OpenaiDeployment
    max_ongoing_requests: {{ app.max_ongoing_requests }}
    autoscaling_config:
      target_ongoing_requests: 64
      min_replicas: {{ app.min_replica }}
      max_replicas: {{ app.max_replica }}
{% elif app.app_type == 'gemini' %}
- name: {{ app.name }}
  route_prefix: /{{ app.name }}
  import_path: matrix.app_server.llm.gemini_proxy:build_app
  args:
    model: {{ app.model_name }}
    api_key: {{ app.api_key }}
  deployments:
  - name: GeminiDeployment
    max_ongoing_requests: {{ app.max_ongoing_requests }}
    autoscaling_config:
      target_ongoing_requests: 64
      min_replicas: {{ app.min_replica }}
      max_replicas: {{ app.max_replica }}
{% elif app.app_type == 'metagen' %}
- name: {{ app.name }}
  route_prefix: /{{ app.name }}
  import_path: matrix.app_server.llm.metagen_proxy:build_app
  args:
    model: {{ app.model_name }}
    access_token: {{ app.access_token }}
  deployments:
  - name: MetagenDeployment
    max_ongoing_requests: {{ app.max_ongoing_requests }}
    autoscaling_config:
      target_ongoing_requests: 64
      min_replicas: {{ app.min_replica }}
      max_replicas: {{ app.max_replica }}
{% elif app.app_type == 'sagemaker' %}
- name: {{ app.name }}
  route_prefix: /{{ app.name }}
  import_path: matrix.app_server.llm.sagemaker_proxy:build_app
  args:
    aws_account: {{ app.aws_account }}
    aws_region: {{ app.aws_region }}
    endpoint_name: {{ app.endpoint_name }}
    model: {{app.model_name}}
  deployments:
  - name: SageMakerDeployment
    max_ongoing_requests: {{ app.max_ongoing_requests }}
    autoscaling_config:
      target_ongoing_requests: 64
      min_replicas: {{ app.min_replica }}
      max_replicas: {{ app.max_replica }}
  {% elif app.app_type == 'code' %}
- name: {{ app.name }}
  route_prefix: /{{ app.name }}
  import_path: matrix.app_server.code.code_execution_app:app
  runtime_env: {}
  args: {}
  deployments:
  - name: CodeExecutionApp
    max_ongoing_requests: 100
    autoscaling_config:
      target_ongoing_requests: 1
      min_replicas: {{ app.min_replica }}
      max_replicas: {{ app.max_replica }}
{% elif app.app_type == 'hello' %}
- name: {{ app.name }}
  route_prefix: /{{ app.name }}
  import_path: matrix.app_server.hello.hello:app
  runtime_env: {}
  args: {}
  deployments:
  - name: HelloDeployment
{% endif %}
"""


def update_vllm_app_params(app: Dict[str, Union[str, int]]):
    model_name = str(app.get("model_name"))
    assert model_name, "please add model_name"
    default_params = llama_model_default_parameters.get(model_name)
    if default_params is None:
        model_size = app.get("model_size")
        assert model_size, f"please specify model size for custom model {model_name}"
        default_model_sizes = {
            p["name"]: p for m, p in llama_model_default_parameters.items()
        }
        default_params = default_model_sizes[model_size].copy()
        assert default_params, f"model_size {model_size} not in {default_model_sizes}"

    app.update({k: v for k, v in default_params.items() if k not in app})  # type: ignore[misc]
    app["use_grpc"] = str(app.get("use_grpc", "false")).lower() == "true"

    return app


def is_sglang_app(app):
    if "deployments" in app:
        return "sglang" in app["deployments"][0]["name"].lower()
    else:
        return False


def write_yaml_file(yaml_file, sglang_yaml_file, update_apps):
    apps, sglang_apps = None, None
    if yaml_file:
        apps = copy.deepcopy(update_apps)
        apps["applications"] = [
            app for app in (apps["applications"] or []) if not is_sglang_app(app)
        ]
        if not apps["applications"]:
            apps["applications"] = None

        yaml_file.seek(0)
        yaml_file.truncate()
        yaml.dump(apps, yaml_file, indent=2, sort_keys=False)
        yaml_file.flush()

    if sglang_yaml_file:
        sglang_apps = copy.deepcopy(update_apps)
        sglang_apps["applications"] = [
            app for app in (sglang_apps["applications"] or []) if is_sglang_app(app)
        ]
        if not sglang_apps["applications"]:
            sglang_apps["applications"] = None

        sglang_yaml_file.seek(0)
        sglang_yaml_file.truncate()
        yaml.dump(sglang_apps, sglang_yaml_file, indent=2, sort_keys=False)
        sglang_yaml_file.flush()

    return apps, sglang_apps


def delete_apps(cluster_info, apps_list: List[Dict[str, Union[str, int]]] | None):
    """delete given apps or everything if None"""
    app_names = None if not apps_list else [app["name"] for app in apps_list]
    os.environ["RAY_ADDRESS"] = get_ray_address(cluster_info)
    apps = list(serve.status().applications.keys())
    deleted = []
    for app in apps:
        if app_names is None or app in app_names:
            serve.delete(app)
            deleted.append(app)
    print(f"Applications deleted {deleted}")

    actors = kill_matrix_actors(
        cluster_info, None if not app_names else str(app_names[0])
    )
    print(f"Actors deleted {actors}")


def get_yaml_for_deployment(
    cluster_info: ClusterInfo,
    action: Action,
    applications: Optional[List[Dict[str, Union[str, int]]]],
    yaml_config: Optional[str],
    existing_apps: List[Dict[str, Union[str, int]]],
):
    """deploy helper function.
    Return modified applications and yaml for deployment"""
    from vllm.engine.arg_utils import AsyncEngineArgs

    temp_dir = cluster_info.temp_dir
    if yaml_config is None:
        assert applications is not None
        yaml_str = Template(common_config).render(
            http_port=cluster_info.http_port,
            grpc_port=cluster_info.grpc_port,
        )

        for app in applications:
            if action == Action.REMOVE:
                assert "name" in app
                found_app = [
                    _app for _app in existing_apps if app["name"] == _app["name"]
                ]
                assert len(found_app) >= 1, "App name {} not found".format(app["name"])
                yaml_str += "\n" + yaml.dump([found_app[0]], indent=2, sort_keys=False)
                continue

            app_type = app.get("app_type", "llm")
            assert app_type in [
                "llm",
                "sglang_llm",
                "code",
                "hello",
                "openai",
                "metagen",
                "sagemaker",
                "gemini",
            ], f"unknown app_type {app_type}"
            app["app_type"] = app_type
            if "min_replica" not in app:
                app["min_replica"] = 1
            if "max_replica" not in app:
                app["max_replica"] = app["min_replica"]

            if app_type in ["llm", "sglang_llm"]:
                unknown = {
                    k: v
                    for k, v in app.items()
                    if k not in non_model_params
                    and not hasattr(AsyncEngineArgs, k.replace("-", "_"))
                    and not hasattr(BaseDeployment, k.replace("-", "_"))
                }
                assert not unknown, f"unknown vllm model args {unknown}"
            else:
                unknown = {k: v for k, v in app.items() if k not in non_model_params}
                assert not unknown, f"unknown {app_type} model args {unknown}"

            if app_type in ["llm", "sglang_llm"]:
                update_vllm_app_params(app)
                yaml_str += Template(vllm_app_template).render(
                    temp_dir=temp_dir, non_model_params=non_model_params, app=app
                )
            elif app_type == "code":
                if "name" not in app:
                    app["name"] = "code"
                yaml_str += Template(other_app_template).render(app=app)
            elif app_type == "openai":
                default_params: Dict[str, Union[str, int]] = {
                    "name": "openai",
                    "model_name": "gpt-4o",
                    "max_ongoing_requests": 100,
                }
                app.update({k: v for k, v in default_params.items() if k not in app})
                assert "api_version" in app, "add api_version to openai app"
                assert "api_endpoint" in app, "add api_endpoint to openai app"
                assert "api_key" in app, "add api_key to openai app"
                yaml_str += Template(other_app_template).render(app=app)
            elif app_type == "metagen":
                default_params = {
                    "name": "metagen",
                    "max_ongoing_requests": 10,
                }
                app.update({k: v for k, v in default_params.items() if k not in app})
                assert "access_token" in app, "add access_token to metagen app"
                yaml_str += Template(other_app_template).render(app=app)
            elif app_type == "sagemaker":
                default_params = {
                    "name": "sagemaker",
                    "max_ongoing_requests": 10,
                }
                app.update({k: v for k, v in default_params.items() if k not in app})

                assert "aws_account" in app, "add aws_account to sagemaker app"
                assert "aws_region" in app, "add aws_region to sagemaker app"
                assert "endpoint_name" in app, "add endpoint_name to sagemaker app"
                assert "model_name" in app, "add model_name to sagemaker app"
                yaml_str += Template(other_app_template).render(app=app)
            elif app_type == "gemini":
                default_params = {
                    "name": "gemini",
                    "max_ongoing_requests": 10,
                }
                app.update({k: v for k, v in default_params.items() if k not in app})
                assert "api_key" in app, "add api_key to gemini app"
                assert "model_name" in app, "add model_name to gemini app"
                yaml_str += Template(other_app_template).render(app=app)
            else:
                assert "name" in app, "add name to app"
                yaml_str += Template(other_app_template).render(app=app)

    else:
        with open(yaml_config, "r") as file:
            template = Template(file.read())
            yaml_str = template.render(
                http_port=cluster_info.http_port, grpc_port=cluster_info.grpc_port
            )
    return yaml_str

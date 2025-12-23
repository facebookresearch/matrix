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
import yaml
from jinja2 import Template

from matrix.app_server.llm.llm_config import llm_model_default_parameters
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
    "anthropic_version",
    "thinking_budget",
    "num_containers_per_replica",
    "ray_resources",  # for ContainerActor
    # Perception encoder and optical flow
    "torch_batch_size",
    # Optical flow
    "return_flow",
    "motion_score",
]

vllm_app_template = """
- name: {{ app.name }}
  route_prefix: /{{ app.name }}
  {% if app.app_type == 'fastgen' %}
  import_path: matrix.app_server.llm.ray_serve_fastgen:build_app
  {% else %}
  import_path: matrix.app_server.llm.ray_serve_vllm:{{ 'build_app_grpc' if app.use_grpc else 'build_app' }}
  {% endif %}
  runtime_env:
    env_vars:
        OUTLINES_CACHE_DIR: {{ temp_dir }}/.outlines
        RAY_DEBUG: legacy
        TIKTOKEN_RS_CACHE_DIR: {{ temp_dir }}
  args:
    model: {{ app.model_name }}
    {% for key, value in app.items() %}
    {% if key not in non_model_params or key in ["ray_resources"]%}
    {{ key }}: {{ 'null' if value is true else value }}
    {% endif %}
    {% endfor %}
  deployments:
  {% if app.use_grpc %}
  - name: GrpcDeployment
  {% elif app.app_type == 'sglang_llm' %}
  - name: SglangDeployment
  {% elif app.app_type == 'fastgen' %}
  - name: FastgenDeployment
  {% else %}
  - name: VLLMDeployment
  {% endif %}
    max_ongoing_requests: {{ app.max_ongoing_requests }}
    autoscaling_config:
      target_ongoing_requests: {{ [ (app.max_ongoing_requests * 0.8) | int , 1 ] | max }}
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
    thinking_budget: {{ app.thinking_budget }}
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
{% elif app.app_type == 'bedrock' %}
- name: {{ app.name }}
  route_prefix: /{{ app.name }}
  import_path: matrix.app_server.llm.bedrock_proxy:build_app
  runtime_env:
    env_vars:
        AWS_ACCESS_KEY_ID: {{ env.AWS_ACCESS_KEY_ID }}
        AWS_SECRET_ACCESS_KEY: {{ env.AWS_SECRET_ACCESS_KEY }}
        AWS_SESSION_TOKEN: {{ env.AWS_SESSION_TOKEN }}
  args:
    aws_region: {{ app.aws_region }}
    model: {{ app.model_name }}
    anthropic_version: {{ app.anthropic_version }}
  deployments:
  - name: BedrockDeployment
    max_ongoing_requests: {{ app.max_ongoing_requests }}
    autoscaling_config:
      target_ongoing_requests: {{ [ (app.max_ongoing_requests * 0.8) | int , 1 ] | max }}
      min_replicas: {{ app.min_replica }}
      max_replicas: {{ app.max_replica }}
{% elif app.app_type == 'llama_api' %}
- name: {{ app.name }}
  route_prefix: /{{ app.name }}
  import_path: matrix.app_server.llm.llama_api_proxy:build_app
  args:
    model: {{ app.model_name }}
    api_key: {{ app.api_key }}
  deployments:
  - name: LlamaApiDeployment
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
  {% elif app.app_type == 'container' %}
- name: {{ app.name }}
  route_prefix: /{{ app.name }}
  import_path: matrix.app_server.container.container_deployment:build_app
  runtime_env: {}
  args:
    num_containers_per_replica: {{ app.max_ongoing_requests }}
    ray_resources: {{ app.ray_resources }}
    name: {{ app.name }}
  deployments:
  - name: ContainerDeployment
    max_ongoing_requests: {{ app.max_ongoing_requests }}
    autoscaling_config:
      target_ongoing_requests: {{ [ (app.max_ongoing_requests * 0.8) | int , 1 ] | max }}
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
{% elif app.app_type == 'perception_encoder' %}
- name: {{ app.name }}
  route_prefix: /{{ app.name }}
  import_path: matrix.app_server.vision.perception_encoder:build_app
  args:
    model_name: {{ app.model_name }}
    {% if app.torch_batch_size is not none %}torch_batch_size: {{ app.torch_batch_size }}{% endif %}
  deployments:
  - name: PerceptionEncoderDeployment
    max_ongoing_requests: 100
    autoscaling_config:
      target_ongoing_requests: 1
      min_replicas: {{ app.min_replica }}
      max_replicas: {{ app.max_replica }}
{% elif app.app_type == 'optical_flow' %}
- name: {{ app.name }}
  route_prefix: /{{ app.name }}
  import_path: matrix.app_server.vision.optical_flow:build_app
  args:
    model_name: {{ app.model_name }}
    {% if app.torch_batch_size is not none %}torch_batch_size: {{ app.torch_batch_size }}{% endif %}
    {% if app.return_flow is not none %}return_flow: {{ app.return_flow }}{% endif %}
    {% if app.motion_score is not none %}motion_score: {{ app.motion_score }}{% endif %}
  deployments:
  - name: OpticalFlowDeployment
    max_ongoing_requests: 100
    autoscaling_config:
      target_ongoing_requests: 1
      min_replicas: {{ app.min_replica }}
      max_replicas: {{ app.max_replica }}
{% endif %}
"""


def update_vllm_app_params(app: Dict[str, Union[str, int]]):
    model_name = str(app.get("model_name"))
    assert model_name, "please add model_name"
    default_params = llm_model_default_parameters.get(model_name)
    if default_params is None:
        model_size = app.get("model_size")
        assert model_size, f"please specify model size for custom model {model_name}"
        default_model_sizes = {
            p["name"]: p for m, p in llm_model_default_parameters.items()
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


def get_app_type(app):
    assert "deployments" in app
    deployment = app["deployments"][0]["name"]
    deploy_type = {
        "ContainerDeployment": "container",
        "CodeExecutionApp": "code",
        "GrpcDeployment": "llm",
        "VLLMDeployment": "llm",
        "SglangDeployment": "sglang_llm",
        "FastgenDeployment": "fastgen",
        "PerceptionEncoderDeployment": "perception_encoder",
        "OpticalFlowDeployment": "optical_flow",
        "LlamaApiDeployment": "llama_api",
    }
    app_type = deploy_type.get(deployment)
    if app_type is None and deployment.endswith("Deployment"):
        app_type = deployment.removesuffix("Deployment").lower()
    return app_type or "unknown"


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
    from ray import serve

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

    from matrix.app_server.llm.ray_serve_vllm import BaseDeployment

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
                "fastgen",
                "code",
                "container",
                "hello",
                "openai",
                "metagen",
                "sagemaker",
                "gemini",
                "bedrock",
                "llama_api",
                "perception_encoder",
                "optical_flow",
            ], f"unknown app_type {app_type}"
            app["app_type"] = app_type
            if "min_replica" not in app:
                app["min_replica"] = 1
            if "max_replica" not in app:
                app["max_replica"] = app["min_replica"]

            if app_type in ["llm", "sglang_llm", "fastgen"]:
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

            if app_type in ["llm", "sglang_llm", "fastgen"]:
                update_vllm_app_params(app)
                yaml_str += Template(vllm_app_template).render(
                    temp_dir=temp_dir, non_model_params=non_model_params, app=app
                )
            elif app_type == "code":
                if "name" not in app:
                    app["name"] = "code"
                yaml_str += Template(other_app_template).render(app=app)
            elif app_type == "container":
                default_params: Dict[str, Union[str, int]] = {
                    "name": "container",
                    "max_ongoing_requests": 32,
                }
                app.update({k: v for k, v in default_params.items() if k not in app})
                yaml_str += Template(other_app_template).render(app=app)
            elif app_type == "openai":
                default_params = {
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
                    "thinking_budget": 1024,
                }
                app.update({k: v for k, v in default_params.items() if k not in app})
                assert "api_key" in app, "add api_key to gemini app"
                assert "model_name" in app, "add model_name to gemini app"
                yaml_str += Template(other_app_template).render(app=app)
            elif app_type == "bedrock":
                default_params = {
                    "name": "bedrock",
                    "max_ongoing_requests": 10,
                    "aws_region": "us-west-2",
                    "anthropic_version": "bedrock-2023-05-31",
                }
                app.update({k: v for k, v in default_params.items() if k not in app})
                assert "model_name" in app, "add model_name to bedrock app"
                env = {k: v for k, v in os.environ.items() if k.startswith("AWS_")}
                yaml_str += Template(other_app_template).render(app=app, env=os.environ)
            elif app_type == "llama_api":
                default_params = {
                    "name": "llama_api",
                    "max_ongoing_requests": 128,
                }
                app.update({k: v for k, v in default_params.items() if k not in app})
                assert "api_key" in app, "add api_key to llama_api app"
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


def validate_applications(applications):
    import fsspec

    for app in applications:
        model = app["model_name"]
        if model.startswith("/"):
            if not os.path.exists(model):
                raise FileNotFoundError(f"{model} does not exists")
            model_config = os.path.join(model, "config.json")
            if not os.path.exists(model_config):
                raise FileNotFoundError(f"{model_config} does not exists")
            with open(model_config, "r", encoding="utf-8") as f:
                config = json.load(f)
                if (
                    app.get("model_size") == "8B"
                    and "Fairseq2LlamaForCausalLM" in config["architectures"]
                ):
                    model_pt = os.path.join(model, "model.pt")
                    if not os.path.exists(model_pt):
                        raise FileNotFoundError(f"{model_pt} does not exists")
        elif model.startswith("s3://"):
            model_config = os.path.join(model, "config.json")
            fs, path = fsspec.core.url_to_fs(model_config)
            if not fs.exists(path):
                raise FileNotFoundError(f"{model_config} does not exists")
            with fsspec.open(model_config, "r", encoding="utf-8") as f:
                # Use the json module to load data from the file-like object
                config = json.load(f)
                if (
                    app.get("model_size") == "8B"
                    and "Fairseq2LlamaForCausalLM" in config["architectures"]
                ):
                    model_pt = os.path.join(model, "model.pt")
                    fs, path = fsspec.core.url_to_fs(model_pt)
                    if not fs.exists(path):
                        raise FileNotFoundError(f"{model_pt} does not exists")
    return True

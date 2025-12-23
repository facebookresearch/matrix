# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import random
import shutil
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Awaitable, Dict, List, Optional, Tuple, Union

import yaml

from matrix.app_server.deploy_utils import (
    delete_apps,
    get_app_type,
    get_yaml_for_deployment,
    is_sglang_app,
    write_yaml_file,
)
from matrix.client.endpoint_cache import EndpointCache
from matrix.common.cluster_info import ClusterInfo, get_head_http_host
from matrix.utils.basics import convert_to_json_compatible, sanitize_app_name
from matrix.utils.os import download_s3_dir, lock_file, run_and_stream, run_async
from matrix.utils.ray import (
    ACTOR_NAME_SPACE,
    Action,
    get_ray_address,
    get_ray_dashboard_address,
    get_serve_applications,
    kill_matrix_actors,
)

logger = logging.getLogger("ray.serve")

DEPLOYMENT_YAML = "deployment.yaml"
DEPLOYMENT_SGLANG_YAML = "deployment_sglang.yaml"


class AppApi:

    def __init__(self, cluster_dir, cluster_info):
        self._cluster_dir = cluster_dir
        self._cluster_info = cluster_info
        self._cluster_id = os.path.basename(cluster_dir)

    def deploy(
        self,
        action: str | Action = Action.REPLACE,
        applications: Optional[List[Dict[str, Union[str, int]]]] = None,
        yaml_config: Optional[str] = None,
    ) -> List[str]:
        """
        Deploy ray serve applications using either a yaml_config file or using the builtin template configured by applications.

        args:
        yaml_config: standard ray serve config file format.
        applications: array of dictionary, eg [{"model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct", "min_replica": 8, "max_replica": 8}]
            or '[{"app_type": "code", "pythonpath": "'"`pwd`/libs/codegen/xlformers/lib"'"}]'

        Return:
        list of app names.
        """
        import ray
        from ray.serve import scripts

        yaml_filepath = str(self._cluster_dir / DEPLOYMENT_YAML)
        sglang_yaml_filepath = str(self._cluster_dir / DEPLOYMENT_SGLANG_YAML)

        if not ray.is_initialized():
            ray.init(
                address=get_ray_address(self._cluster_info),
                ignore_reinit_error=True,
                log_to_driver=False,
            )

        temp_dir = self._cluster_info.temp_dir
        assert temp_dir, "head temp_dir is None"
        os.environ["OUTLINES_CACHE_DIR"] = os.path.join(temp_dir, ".outlines")

        assert yaml_config is None or os.path.exists(
            yaml_config
        ), f"{yaml_config} not found"
        assert (applications is None) != (
            yaml_config is None
        ), "provide a yaml_config file or the applications"
        try:
            action = Action(action) if isinstance(action, str) else action
        except ValueError:
            raise ValueError(
                f"Invalid action '{action}', expected one of {[a.value for a in Action]}"
            )
        if action in [Action.ADD, Action.REPLACE]:
            for app in applications or []:
                if str(app.get("model_name", "")).startswith("s3://"):
                    cache_dir = os.environ.get(
                        "MATRIX_CACHE", os.path.expanduser("~/.cache/matrix")
                    )
                    cache_dir = os.path.join(cache_dir, self._cluster_id, "models")
                    s3_dir = app["model_name"]
                    logger.info(f"Download {s3_dir} under {cache_dir}")
                    downloaded, dest_dir = download_s3_dir(
                        str(s3_dir), cache_dir, 3, "*rank_*.pt"
                    )
                    if not downloaded:
                        raise ValueError(f"Can not read {s3_dir}")
                    app["model_name"] = dest_dir

        with lock_file(yaml_filepath, "a+", timeout=10) as yaml_file:
            with lock_file(sglang_yaml_filepath, "a+", timeout=10) as sglang_yaml_file:
                yaml_file.seek(0)
                old = yaml.safe_load(yaml_file)
                if old is None:
                    old_apps: List[Dict[str, Union[str, int]]] = []
                else:
                    old_apps = old["applications"] or []
                sglang_yaml_file.seek(0)
                sglang_old = yaml.safe_load(sglang_yaml_file)
                if sglang_old is None:
                    sglang_old_apps: List[Dict[str, Union[str, int]]] = []
                else:
                    sglang_old_apps = sglang_old["applications"] or []
                existing_apps = old_apps + sglang_old_apps
                existing_app_names = [app["name"] for app in existing_apps]
                assert applications is not None
                for _i in range(len(applications) - 1, -1, -1):
                    app = applications[_i]
                    if action == Action.ADD and app.get("name") is None:
                        hex_hash = hashlib.sha256(
                            (str(app.get("model_name")) + str(time.time())).encode()
                        ).digest()
                        name = base64.b32encode(hex_hash).decode()[:8]
                        app["name"] = name

                    if app.get("name") is not None:
                        sanitized = sanitize_app_name(str(app["name"]))
                        if sanitized != app["name"]:
                            logger.info(
                                f"Sanitized app name {app['name']} -> {sanitized}"
                            )
                            app["name"] = sanitized

                    found = app.get("name") in existing_app_names
                    if found and action == Action.ADD:
                        logger.warning(
                            f"Ignore adding app {app}, already exist in {existing_app_names}"
                        )
                        del applications[_i]
                    elif not found and action == Action.REMOVE:
                        logger.warning(
                            f"Ignore removing app {app}, does not exist in {existing_app_names}"
                        )
                        del applications[_i]

                yaml_str = get_yaml_for_deployment(
                    self._cluster_info, action, applications, yaml_config, existing_apps
                )
                update_apps = yaml.safe_load(yaml_str)

                if update_apps["applications"] is None:
                    if action == Action.REPLACE:
                        # special case of remove everything
                        delete_apps(self._cluster_info, None)
                        write_yaml_file(yaml_file, sglang_yaml_file, update_apps)
                else:
                    if action == Action.REMOVE:
                        delete_apps(self._cluster_info, update_apps["applications"])
                        remove_names = [
                            app["name"] for app in update_apps["applications"]
                        ]
                        old_apps = [
                            app for app in old_apps if app["name"] not in remove_names
                        ]
                        sglang_old_apps = [
                            app
                            for app in sglang_old_apps
                            if app["name"] not in remove_names
                        ]
                        remaining = old or sglang_old
                        remaining["applications"] = old_apps + sglang_old_apps
                        write_yaml_file(yaml_file, sglang_yaml_file, remaining)
                    else:
                        # separate deploy for serve and sglang
                        sglang_apps = [
                            app
                            for app in update_apps["applications"]
                            if is_sglang_app(app)
                        ]
                        if sglang_apps:
                            from matrix.app_server.llm import deploy_sglang_app

                            assert (
                                len(update_apps["applications"]) == 1
                            ), "only support 1 sglang app"
                            assert (
                                applications is not None and len(applications) == 1
                            ), "sglang does not support yaml deploy"
                            write_yaml_file(None, sglang_yaml_file, update_apps)
                            kill_matrix_actors(self._cluster_info)
                            deploy_sglang_app.deploy_app(
                                self._cluster_dir, self._cluster_info, applications[0]
                            )
                        else:
                            if action == Action.ADD:
                                # disjoint
                                old_app_names = [app["name"] for app in existing_apps]
                                new_app_names = [
                                    app["name"] for app in update_apps["applications"]
                                ]
                                duplicates = set(old_app_names) & set(new_app_names)
                                assert (
                                    not duplicates
                                ), f"Add to existing apps {duplicates}"

                                update_apps["applications"].extend(existing_apps)
                            serve_apps, _ = write_yaml_file(
                                yaml_file, None, update_apps
                            )
                            assert serve_apps["applications"]
                            scripts.deploy(
                                [
                                    "--address",
                                    get_ray_dashboard_address(self._cluster_info),
                                    yaml_file.name,
                                ],
                                standalone_mode=False,
                            )
                return [app["name"] for app in (update_apps.get("applications") or [])]

    def status(self, replica):
        """Print out Serve applications and matrix actors."""
        import ray
        from ray.serve.context import _get_global_client

        results = []
        ray_dashboard_url = get_ray_dashboard_address(self._cluster_info)
        serve_status = run_and_stream(
            {},
            " ".join(["serve", "status", "--address", ray_dashboard_url]),
            blocking=True,
            return_stdout_lines=1000,
        )
        results.extend(
            serve_status.get("stdout", serve_status.get("error", "")).split("\n")
        )

        actor_status = run_and_stream(
            {},
            " ".join(
                [
                    "ray",
                    "list",
                    "actors",
                    "--address",
                    ray_dashboard_url,
                    "--filter",
                    "ray_namespace=matrix",
                    "--filter",
                    "state!=DEAD",
                    "--limit",
                    "10000",
                ]
            ),
            blocking=True,
            return_stdout_lines=1000,
        )
        results.extend(
            actor_status.get("stdout", actor_status.get("error", "")).split("\n")
        )
        if replica:
            results.append("\n\nReplica: " + "-" * 8)
            os.environ["RAY_ADDRESS"] = get_ray_address(self._cluster_info)
            _client = _get_global_client()
            assert _client is not None
            replicas = ray.get(
                _client._controller._all_running_replicas.remote()  # type: ignore[attr-defined]
            )  # type: ignore[union-attr]
            json_compatible_replicas = convert_to_json_compatible(replicas)
            results.append(json.dumps(json_compatible_replicas, indent=2))
        return results

    def _read_deployment(self, app_name, deployment_file, model_name=None):

        yaml_config = str(self._cluster_dir / deployment_file)
        if not os.path.exists(yaml_config):
            logger.debug(f"config does not exist {yaml_config}")
            return None, None
        with open(yaml_config, "r") as file:
            data = yaml.safe_load(file)
        if data is None:
            logger.debug(f"empty config {yaml_config}")
            return None, None

        app = [
            a
            for a in (data["applications"] or [])
            if (
                (app_name and a["name"] == app_name)
                or (model_name and a["args"]["model"] == model_name)
            )
        ]
        if len(app) == 1:
            return app[0], data
        else:
            return None, None

    def get_app_metadata(
        self,
        app_name: str,
        endpoint_ttl_sec: int = 5,
        model_name: Optional[str] = None,
        head_only: bool = False,
    ) -> Dict[str, Any]:
        """Return app's metadata, such as port, head, workers etc"""

        http_port, grpc_port = None, None

        serve_app = True
        app, full_json = self._read_deployment(app_name, DEPLOYMENT_YAML, model_name)
        if app is None:
            logger.info("Nothing found. try sglang deployment")
            serve_app = False
            app, full_json = self._read_deployment(
                app_name, DEPLOYMENT_SGLANG_YAML, model_name
            )

        assert app, f"uknown app_name {app_name} within deployment {app}"
        http_port = full_json["http_options"]["port"]
        grpc_port = full_json["grpc_options"]["port"]

        prefix = app["route_prefix"].strip("/")  # type: ignore
        model = app["args"].get("model")  # type: ignore
        served_model_name = app["args"].get("served_model_name")  # type: ignore
        deployment_name = app["deployments"][0]["name"]  # type: ignore
        use_grpc = "GrpcDeployment" in deployment_name

        if serve_app:
            if (
                "code" in deployment_name.lower()
                or "hello" in deployment_name.lower()
                or "container" in deployment_name.lower()
                or "perception_encoder" in deployment_name.lower()
                or "optical" in deployment_name.lower()
            ):
                endpoint_template = f"http://{{host}}:{http_port}/{prefix}"
            else:
                endpoint_template = (
                    f"http://{{host}}:{http_port}/{prefix}/v1"
                    if not use_grpc
                    else f"{{host}}:{grpc_port}"
                )
        else:
            endpoint_template = (
                f"http://{{host}}:{self._cluster_info.sglang_http_port}/v1"
            )
        metadata = {
            "name": app_name,
            "http_port": http_port,
            "grpc_port": grpc_port,
            "route_prefix": prefix,
            "model_name": served_model_name if served_model_name else model,
            "deployment_name": deployment_name,
            "use_grpc": use_grpc,
            "endpoint_template": endpoint_template,
            "app_type": get_app_type(app),
            "args": app["args"],
        }

        head = metadata["endpoint_template"].format(
            host=get_head_http_host(self._cluster_info)
        )
        if head_only:

            async def dummy_updater():
                return head

            endpoint_cache = dummy_updater
            workers = []
        else:
            endpoint_cache = EndpointCache(
                self._cluster_info,
                metadata["name"],
                metadata["endpoint_template"],
                ttl=endpoint_ttl_sec,
                serve_app=serve_app,
            )
            workers = run_async(endpoint_cache())
        metadata["endpoints"] = {
            "head": head,
            "workers": workers,
            "updater": endpoint_cache,
        }

        return metadata

    def inference(
        self,
        app_name: str,
        output_jsonl: str,
        input_jsonls: str | None = None,
        input_hf_dataset: str | None = None,
        hf_dataset_split: str = "train",
        load_balance: bool = True,
        **kwargs,
    ):
        """Run LLM inference.

        The input can be provided either as JSONL files via ``input_jsonls`` or
        fetched directly from a Hugging Face dataset using ``input_hf_dataset``
        and ``hf_dataset_split``.
        """

        metadata = self.get_app_metadata(app_name)
        assert self._cluster_info.hostname
        local_mode = self._cluster_info.executor == "local"

        async def get_one_endpoint() -> str:
            if not load_balance:
                return metadata["endpoint_template"].format(
                    host="localhost" if local_mode else self._cluster_info.hostname
                )
            else:
                ips = await metadata["endpoints"]["updater"]()
                assert ips
                host = random.choice(ips)
                return host

        app_type = metadata["app_type"]
        if app_type in [
            "llm",
            "sglang_llm",
            "fastgen",
            "openai",
            "metagen",
            "sagemaker",
            "gemini",
            "bedrock",
            "llama_api",
        ]:
            from matrix.client.query_llm import main as query_llm

            return asyncio.run(
                query_llm(
                    get_one_endpoint,
                    output_jsonl,
                    input_jsonls,
                    model=metadata["model_name"],
                    app_name=metadata["name"],
                    input_hf_dataset=input_hf_dataset,
                    hf_dataset_split=hf_dataset_split,
                    **kwargs,
                )
            )
        elif app_type == "code":
            from matrix.client.execute_code import CodeExcutionClient

            client = CodeExcutionClient(get_one_endpoint)
            assert input_jsonls is not None, "input_jsonls is required for code apps"
            return asyncio.run(
                client.execute_code(
                    output_jsonl,
                    input_jsonls,
                    **kwargs,
                )
            )
        elif app_type in {"perception_encoder", "optical_flow"}:
            from matrix.client.process_vision_data import VisionClient

            vision_client = VisionClient(get_one_endpoint)
            assert input_jsonls is not None, "input_jsonls is required for vision apps"
            return asyncio.run(
                vision_client.inference(
                    output_jsonl,
                    input_jsonls,
                    **kwargs,
                )
            )
        else:
            raise ValueError(f"app_type {app_type} is not supported.")

    def app_status(self, app_name: str) -> str:
        """The current status of the application.

        As from Ray
        class ApplicationStatus(str, Enum):
            NOT_STARTED = "NOT_STARTED"
            DEPLOYING = "DEPLOYING"
            DEPLOY_FAILED = "DEPLOY_FAILED"
            RUNNING = "RUNNING"
            UNHEALTHY = "UNHEALTHY"
            DELETING = "DELETING"
        """
        import ray

        app, _full_json = self._read_deployment(app_name, DEPLOYMENT_YAML)
        if app is None:
            serve_app = False
            app, _full_json = self._read_deployment(app_name, DEPLOYMENT_SGLANG_YAML)
        else:
            serve_app = True
        assert app, f"uknown app_name {app_name} within deployment {app}"

        if serve_app:
            url = get_ray_dashboard_address(self._cluster_info)
            apps = get_serve_applications(url)
            found_app = apps["applications"].get(app_name)
            if found_app is None:
                return "NOT_STARTED"
            else:
                return found_app["status"]
        else:
            try:
                min_replica = app["deployments"][0]["autoscaling_config"]["min_replica"]
                router_actor = ray.get_actor(f"{app_name}_router", ACTOR_NAME_SPACE)
                is_running, replicas = ray.get(
                    [
                        router_actor.is_running.remote(),
                        router_actor.get_running_replicas.remote(),
                    ]
                )
                # todo: also check the actor state
                if not is_running:
                    return "DEPLOYING"
                elif len(replicas) < min_replica:
                    return "DEPLOYING"
                else:
                    return "RUNNING"
            except:
                return "NOT_STARTED"

    def app_cleanup(self, app_name: str) -> str:
        """A helper function to cleanup for stateful services, eg containers maybe be dangling due to exception etc."""
        from matrix.client.container_client import ContainerClient

        metadata = self.get_app_metadata(app_name)
        assert metadata["app_type"] == "container"
        base_url = metadata["endpoints"]["head"]
        client = ContainerClient(base_url)
        result = run_async(client.release_all_containers())
        run_async(client.close())
        return result

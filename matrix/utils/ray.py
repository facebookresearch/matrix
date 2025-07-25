# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import subprocess
from enum import Enum

import aiohttp

from matrix.common.cluster_info import ClusterInfo
from matrix.utils.http import fetch_url_sync

ACTOR_NAME_SPACE = "matrix"


class Action(Enum):
    REPLACE = "replace"
    ADD = "add"
    REMOVE = "remove"


def get_ray_address(cluster_info: ClusterInfo) -> str:
    return f"ray://{cluster_info.hostname}:{cluster_info.client_server_port}"


def get_ray_dashboard_address(cluster_info: ClusterInfo) -> str:
    return f"http://{cluster_info.hostname}:{cluster_info.dashboard_port}"


def get_matrix_actors(cluster_info, prefix=None, include_pending=False):
    # Run the Ray status command to get actor information, workaround for double init
    filter = "state!=DEAD" if include_pending else "state=ALIVE"
    result = subprocess.run(
        [
            "ray",
            "list",
            "actors",
            "--format=json",
            f"--address={get_ray_address(cluster_info)}",
            "--filter",
            "ray_namespace=matrix",
            "--filter",
            filter,
            "--limit",
            "10000",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        try:
            actors = json.loads(result.stdout)
            if prefix:
                actors = [ac for ac in actors if ac["name"].startswith(prefix)]
            return actors
        except:
            return []
    else:
        print("Error running Ray list actors:", result.stderr)
        return []


def get_ray_head_node():
    import ray

    for node in ray.nodes():
        if node["Alive"] and node["Resources"].get("node:__internal_head__"):
            return node
    raise Exception("no head")


def kill_matrix_actors(cluster_info, prefix: str | None = None):
    # todo: also delete task?
    import ray

    changed = True
    deleted = []
    while changed:
        actors = get_matrix_actors(cluster_info, include_pending=True)
        actors = [
            ac
            for ac in actors
            if ac["ray_namespace"] == ACTOR_NAME_SPACE
            and (prefix is None or ac["name"].startswith(prefix))
            and "system." not in ac["name"]
        ]
        names = [actor["name"] for actor in actors]
        handles = [ray.get_actor(name, ACTOR_NAME_SPACE) for name in names]
        [handle.kill.remote() for handle in handles]
        [ray.kill(handle) for handle in handles]
        deleted.extend(names)
        changed = len(handles) > 0
    return deleted


def init_ray_if_necessary(cluster_info: ClusterInfo):
    import ray

    ray_address = get_ray_address(cluster_info)
    if not ray.is_initialized():
        ray.init(
            address=ray_address,
            ignore_reinit_error=True,
        )


def get_serve_applications(ray_http_address):
    status, content = fetch_url_sync(
        ray_http_address + "/api/serve/applications/",
        headers={"Accept": "application/json"},
    )
    if status is not None and status == 200:
        return json.loads(content)
    else:
        return []


def status_is_success(app_status: str) -> bool:
    return app_status == "RUNNING"


def status_is_failure(app_status: str) -> bool:
    return app_status in ["DEPLOY_FAILED", "DELETING"]


def status_is_pending(app_status: str) -> bool:
    return app_status in ["NOT_STARTED", "DEPLOYING", "UNHEALTHY"]

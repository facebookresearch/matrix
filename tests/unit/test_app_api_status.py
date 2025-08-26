# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import sys
import types
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from matrix.app_server.app_api import AppApi
from matrix.common.cluster_info import ClusterInfo


class DummyDeploymentID:
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return f"DummyDeploymentID({self.name})"


@dataclass
class DummyReplica:
    id: int
    name: str


@patch(
    "matrix.app_server.app_api.run_and_stream",
    return_value={"stdout": "", "exit_code": 0},
)
@patch("matrix.app_server.app_api.get_ray_address", return_value="ray://host:10001")
@patch(
    "matrix.app_server.app_api.get_ray_dashboard_address",
    return_value="http://host:8265",
)
def test_status_replica_json_serializable(
    mock_dashboard_addr,
    mock_ray_addr,
    mock_run_and_stream,
    tmp_path,
):
    replica_info = DummyReplica(id=1, name="r1")
    deployment_id = DummyDeploymentID("app")
    replicas = {deployment_id: replica_info}

    controller = MagicMock()
    controller._all_running_replicas.remote.return_value = replicas
    client = MagicMock()
    client._controller = controller

    dummy_context = types.SimpleNamespace(_get_global_client=lambda: client)
    dummy_serve = types.SimpleNamespace(context=dummy_context)
    dummy_ray = types.SimpleNamespace(get=lambda x: replicas, serve=dummy_serve)
    with patch.dict(
        sys.modules,
        {
            "ray": dummy_ray,
            "ray.serve": dummy_serve,
            "ray.serve.context": dummy_context,
        },
    ):
        cluster_info = ClusterInfo(
            hostname="host", client_server_port=10001, dashboard_port=8265
        )
        api = AppApi(str(tmp_path), cluster_info)

        results = api.status(replica=True)
        json_output = results[-1]
        data = json.loads(json_output)
        assert "DummyDeploymentID(app)" in data
        assert data["DummyDeploymentID(app)"]["id"] == 1

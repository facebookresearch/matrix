# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from matrix.common.cluster_info import ClusterInfo
from matrix.utils import ray as ray_utils


def test_get_ray_addresses():
    info = ClusterInfo(hostname="host", client_server_port=10001, dashboard_port=8265)
    assert ray_utils.get_ray_address(info) == "ray://host:10001"
    assert ray_utils.get_ray_dashboard_address(info) == "http://host:8265"


def test_status_helpers():
    assert ray_utils.status_is_success("RUNNING")
    for status in ["DEPLOY_FAILED", "DELETING"]:
        assert ray_utils.status_is_failure(status)
    for status in ["NOT_STARTED", "DEPLOYING", "UNHEALTHY"]:
        assert ray_utils.status_is_pending(status)

    for fn in [
        ray_utils.status_is_success,
        ray_utils.status_is_failure,
        ray_utils.status_is_pending,
    ]:
        assert not fn("UNKNOWN")


def test_get_matrix_actors(monkeypatch):
    info = ClusterInfo(hostname="h", client_server_port=1, dashboard_port=2)
    actors = [
        {"name": "a1", "ray_namespace": "matrix"},
        {"name": "b1", "ray_namespace": "matrix"},
    ]
    result_ok = SimpleNamespace(returncode=0, stdout=json.dumps(actors))
    with patch("subprocess.run", return_value=result_ok):
        assert ray_utils.get_matrix_actors(info, prefix="a") == [
            {"name": "a1", "ray_namespace": "matrix"}
        ]

    result_bad_json = SimpleNamespace(returncode=0, stdout="not json")
    with patch("subprocess.run", return_value=result_bad_json):
        assert ray_utils.get_matrix_actors(info) == []

    result_error = SimpleNamespace(returncode=1, stderr="boom", stdout="")
    with patch("subprocess.run", return_value=result_error):
        assert ray_utils.get_matrix_actors(info) == []


def test_get_serve_applications(monkeypatch):
    payload = [{"name": "app"}]
    monkeypatch.setattr(
        ray_utils,
        "fetch_url_sync",
        lambda url, headers=None: (200, json.dumps(payload)),
    )
    assert ray_utils.get_serve_applications("http://ray") == payload

    monkeypatch.setattr(
        ray_utils, "fetch_url_sync", lambda url, headers=None: (500, "error")
    )
    assert ray_utils.get_serve_applications("http://ray") == []


@pytest.mark.asyncio
async def test_ray_get_async_single():
    fake_ray = SimpleNamespace(get=lambda ref, timeout=None: ref * 2)
    with patch.dict(sys.modules, {"ray": fake_ray}):
        assert await ray_utils.ray_get_async(3) == 6


@pytest.mark.asyncio
async def test_ray_get_async_list():
    def fake_wait(refs, num_returns, timeout):
        return refs, []

    fake_ray = SimpleNamespace(wait=fake_wait, get=lambda ref: ref * 2)
    with patch.dict(sys.modules, {"ray": fake_ray}):
        assert await ray_utils.ray_get_async([1, 2]) == [2, 4]


@pytest.mark.asyncio
async def test_ray_get_async_timeout():
    def fake_wait(refs, num_returns, timeout):
        return [], refs

    fake_ray = SimpleNamespace(wait=fake_wait, get=lambda ref, timeout=None: ref)
    with patch.dict(sys.modules, {"ray": fake_ray}):
        with pytest.raises(TimeoutError):
            await ray_utils.ray_get_async([1, 2], timeout=0)


def test_kill_matrix_actors(monkeypatch):
    info = ClusterInfo(hostname="h", client_server_port=1, dashboard_port=2)

    actor_seq = iter(
        [
            [
                {"name": "a1", "ray_namespace": "matrix"},
                {"name": "system.actor", "ray_namespace": "matrix"},
            ],
            [],
        ]
    )

    monkeypatch.setattr(
        ray_utils, "get_matrix_actors", lambda *a, **kw: next(actor_seq)
    )

    calls: list[str] = []

    class Handle:
        def __init__(self, name):
            self.name = name

            class K:
                def __init__(self, n):
                    self.n = n

                def remote(self):
                    calls.append(f"kill.remote:{self.n}")

            self.kill = K(name)

    def fake_get_actor(name, ns):
        return Handle(name)

    def fake_kill(handle):
        calls.append(f"ray.kill:{handle.name}")

    fake_ray = SimpleNamespace(get_actor=fake_get_actor, kill=fake_kill)
    with patch.dict(sys.modules, {"ray": fake_ray}):
        deleted = ray_utils.kill_matrix_actors(info, prefix="a")

    assert deleted == ["a1"]
    assert calls == ["kill.remote:a1", "ray.kill:a1"]

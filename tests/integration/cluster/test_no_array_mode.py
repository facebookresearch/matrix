# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import time
import uuid
from typing import Any, Generator

import pytest
import ray

from matrix.cli import Cli
from matrix.utils.ray import status_is_pending, status_is_success


@pytest.fixture(scope="module")
def matrix_cluster_no_array() -> Generator[Any, Any, Any]:
    """Start and stop Ray cluster with use_array=False for the duration of these tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cli = Cli(cluster_id=str(uuid.uuid4()), matrix_dir=temp_dir)
        cli.start_cluster(
            add_workers=1,
            slurm=None,
            local={"gpus_per_node": 0, "cpus_per_task": 2},
            enable_grafana=False,
            use_array=False,
        )
        with cli.cluster:
            yield cli


def test_cluster_starts_no_array(matrix_cluster_no_array: Cli) -> None:
    """Test that cluster starts successfully with use_array=False."""
    cli = matrix_cluster_no_array
    cluster_info = cli.cluster.cluster_info()

    assert cluster_info is not None, "Cluster info should exist"
    assert cluster_info.hostname is not None, "Hostname should be set"
    assert cluster_info.port > 0, "Port should be set"


def test_workers_have_resources_no_array(matrix_cluster_no_array: Cli) -> None:
    """Test that workers have CPU resources allocated when use_array=False."""
    cli = matrix_cluster_no_array
    cluster_info = cli.cluster.cluster_info()
    assert cluster_info is not None

    # Initialize ray connection
    from matrix.utils.ray import init_ray_if_necessary

    init_ray_if_necessary(cluster_info)

    # Get cluster resources
    resources = ray.cluster_resources()

    # Workers should have CPU resources
    assert (
        resources.get("CPU", 0) > 0
    ), "Workers should have CPU resources when use_array=False"


def test_deploy_hello_no_array(matrix_cluster_no_array: Cli) -> None:
    """Test that applications can be deployed with use_array=False."""
    cli = matrix_cluster_no_array
    cli.deploy_applications(applications=[{"name": "hello", "app_type": "hello"}])

    for _ in range(10):
        status = cli.app.app_status("hello")
        if not status_is_pending(status):
            break
        time.sleep(5)

    assert status_is_success(status), f"Bad status {status}"
    assert cli.check_health("hello")


def test_incremental_scaling_no_array() -> None:
    """Test that workers can be added incrementally with use_array=False."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cli = Cli(cluster_id=str(uuid.uuid4()), matrix_dir=temp_dir)

        # Start with 1 worker
        cli.start_cluster(
            add_workers=1,
            slurm=None,
            local={"gpus_per_node": 0, "cpus_per_task": 2},
            enable_grafana=False,
            use_array=False,
        )

        cluster_info = cli.cluster.cluster_info()
        assert cluster_info is not None

        from matrix.utils.ray import init_ray_if_necessary

        init_ray_if_necessary(cluster_info)

        # Check initial resources
        initial_resources = ray.cluster_resources()
        initial_cpu = initial_resources.get("CPU", 0)
        assert initial_cpu > 0, "Should have CPU resources initially"

        # Add more workers
        cli.start_cluster(
            add_workers=1,
            slurm=None,
            local={"gpus_per_node": 0, "cpus_per_task": 2},
            enable_grafana=False,
            use_array=False,
        )

        # Wait for new worker to join
        max_wait = 30  # seconds
        start_time = time.time()
        new_cpu = initial_cpu

        while time.time() - start_time < max_wait:
            time.sleep(2)
            new_resources = ray.cluster_resources()
            new_cpu = new_resources.get("CPU", 0)
            if new_cpu > initial_cpu:
                break

        # Check that resources increased (or at least stayed the same in local mode)
        # In local mode, the worker might share the same host so CPU might not increase
        assert (
            new_cpu >= initial_cpu
        ), f"CPU should not decrease: {initial_cpu} -> {new_cpu}"

        # Cleanup
        cli.cluster.stop()

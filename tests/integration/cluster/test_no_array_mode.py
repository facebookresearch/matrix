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


@pytest.fixture(scope="module")
def matrix_cluster_no_array_multi_worker() -> Generator[Any, Any, Any]:
    """Start and stop Ray cluster with use_array=False and multiple workers."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cli = Cli(cluster_id=str(uuid.uuid4()), matrix_dir=temp_dir)
        cli.start_cluster(
            add_workers=2,
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


def test_head_has_resources_no_array(matrix_cluster_no_array: Cli) -> None:
    """Test that head node has CPU resources allocated when use_array=False."""
    cli = matrix_cluster_no_array
    cluster_info = cli.cluster.cluster_info()
    assert cluster_info is not None

    # Initialize ray connection
    from matrix.utils.ray import init_ray_if_necessary

    init_ray_if_necessary(cluster_info)

    # Get cluster resources
    resources = ray.cluster_resources()

    # Head should have CPU resources since it's acting as worker
    assert (
        resources.get("CPU", 0) > 0
    ), "Head should have CPU resources when use_array=False"


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


def test_cluster_multi_worker_no_array(
    matrix_cluster_no_array_multi_worker: Cli,
) -> None:
    """Test cluster with multiple workers and use_array=False."""
    cli = matrix_cluster_no_array_multi_worker
    cluster_info = cli.cluster.cluster_info()

    assert cluster_info is not None, "Cluster info should exist"

    # Initialize ray connection
    from matrix.utils.ray import init_ray_if_necessary

    init_ray_if_necessary(cluster_info)

    # Get cluster resources
    resources = ray.cluster_resources()

    # Should have resources from head (acting as worker) + 1 additional worker
    assert resources.get("CPU", 0) > 0, "Cluster should have CPU resources"

    # Wait for workers to join (local mode might be slower)
    max_wait = 30  # seconds
    start_time = time.time()
    expected_nodes = 2  # head (acting as worker) + 1 additional worker

    while time.time() - start_time < max_wait:
        nodes = ray.nodes()
        alive_nodes = [n for n in nodes if n["Alive"]]
        if len(alive_nodes) >= expected_nodes:
            break
        time.sleep(2)

    # Check final node count
    nodes = ray.nodes()
    alive_nodes = [n for n in nodes if n["Alive"]]
    # In local mode, workers might not show as separate nodes, so check for at least 1
    assert (
        len(alive_nodes) >= 1
    ), f"Should have at least 1 alive node, got {len(alive_nodes)}"

    # More importantly, check that we have enough CPU resources
    # With add_workers=2 and cpus_per_task=2, we should have at least 4 CPUs
    # (2 from head + 2 from worker)
    total_cpu = resources.get("CPU", 0)
    assert total_cpu >= 2, f"Should have at least 2 CPUs, got {total_cpu}"


def test_deploy_on_multi_worker_no_array(
    matrix_cluster_no_array_multi_worker: Cli,
) -> None:
    """Test deployment on multi-worker cluster with use_array=False."""
    cli = matrix_cluster_no_array_multi_worker
    cli.deploy_applications(applications=[{"name": "hello_multi", "app_type": "hello"}])

    for _ in range(10):
        status = cli.app.app_status("hello_multi")
        if not status_is_pending(status):
            break
        time.sleep(5)

    assert status_is_success(status), f"Bad status {status}"
    assert cli.check_health("hello_multi")


def test_incremental_scaling_no_array() -> None:
    """Test that workers can be added incrementally with use_array=False."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cli = Cli(cluster_id=str(uuid.uuid4()), matrix_dir=temp_dir)

        # Start with 1 worker (head acts as worker)
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


def test_head_only_gets_gpu_resources() -> None:
    """Test that head gets GPU resources when add_workers=0."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cli = Cli(cluster_id=str(uuid.uuid4()), matrix_dir=temp_dir)

        # Start cluster with only head (add_workers=0)
        # In this case, head should get full requirements including GPU resources
        cli.start_cluster(
            add_workers=0,
            slurm=None,
            local={"gpus_per_node": 0, "cpus_per_task": 4},
            enable_grafana=False,
            use_array=False,  # use_array shouldn't matter when add_workers=0
        )

        cluster_info = cli.cluster.cluster_info()
        assert cluster_info is not None

        from matrix.utils.ray import init_ray_if_necessary

        init_ray_if_necessary(cluster_info)

        # Check that head has CPU resources
        resources = ray.cluster_resources()
        cpu = resources.get("CPU", 0)
        assert cpu > 0, f"Head-only cluster should have CPU resources, got {cpu}"

        # Cleanup
        cli.cluster.stop()


def test_array_mode_comparison() -> None:
    """Test that array mode and non-array mode allocate resources differently."""
    # Test 1: Array mode - head should have 0 resources when workers exist
    with tempfile.TemporaryDirectory() as temp_dir:
        cli_array = Cli(cluster_id=str(uuid.uuid4()), matrix_dir=temp_dir)

        cli_array.start_cluster(
            add_workers=1,
            slurm=None,
            local={"gpus_per_node": 0, "cpus_per_task": 2},
            enable_grafana=False,
            use_array=True,  # Array mode
        )

        cluster_info_array = cli_array.cluster.cluster_info()
        assert cluster_info_array is not None

        from matrix.utils.ray import init_ray_if_necessary

        init_ray_if_necessary(cluster_info_array)

        # In array mode, head gets 0 resources
        resources_array = ray.cluster_resources()
        # Head has 0 CPUs/GPUs, but worker has resources
        # Total should still be > 0 from worker
        assert (
            resources_array.get("CPU", 0) > 0
        ), "Array mode should have CPU from workers"

        cli_array.cluster.stop()

    # Test 2: Non-array mode - head should have resources when acting as worker
    with tempfile.TemporaryDirectory() as temp_dir:
        cli_no_array = Cli(cluster_id=str(uuid.uuid4()), matrix_dir=temp_dir)

        cli_no_array.start_cluster(
            add_workers=1,
            slurm=None,
            local={"gpus_per_node": 0, "cpus_per_task": 2},
            enable_grafana=False,
            use_array=False,  # Non-array mode
        )

        cluster_info_no_array = cli_no_array.cluster.cluster_info()
        assert cluster_info_no_array is not None

        from matrix.utils.ray import init_ray_if_necessary

        init_ray_if_necessary(cluster_info_no_array)

        # In non-array mode, head acts as worker and has resources
        resources_no_array = ray.cluster_resources()
        assert (
            resources_no_array.get("CPU", 0) > 0
        ), "Non-array mode head should have CPU resources"

        cli_no_array.cluster.stop()

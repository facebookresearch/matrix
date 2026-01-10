#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to start/reuse a cluster and deploy multiple models.
Waits until all models are RUNNING or exits with timeout.

Example usage:
    python -m matrix.scripts.deploy_models \
    --applications "[{'model_name': '/checkpoint/data/shared/pretrained-llms/Qwen3-30B-A3B-Instruct-2507', 'model_size': 'Qwen3-32B', 'name
': 'qwen3', 'min_replica': 4}, {'model_name': '/checkpoint/data/shared/pretrained-llms/Qwen3-VL-30B-A3B-Instruct', 'model_size': 'Qwen3-VL-30B-A3B-Instruct', 'name': 'qwen3vl', 'min_replica': 4}]" \
    --num_workers=2 --slurm "{'account': 'data', 'qos': 'h100_lowest'}"
"""

import json
import time
import typing as tp

import fire

import matrix
from matrix.app_server import app_api


def main(
    applications: tp.List[tp.Dict[str, tp.Union[str, int]]],
    num_workers: int,
    slurm: tp.Dict[str, tp.Union[str, int]],
    cluster_id: str | None = None,
    timeout: int = 1800,  # 30 minutes default timeout
):
    """
    Start or reuse a cluster and deploy multiple models, waiting until all are RUNNING.

    Args:
        cluster_id: Unique identifier for the cluster
        applications: JSON string or list of application configurations. Each dict should contain:
                     - model_name: Path to the model
                     - min_replica: Minimum number of replicas
                     - name: Application name
                     - model_size: Model size (e.g., "8B")
                     - use_grpc: (optional) Whether to use gRPC
        num_workers: Number of worker nodes (script will only add workers if current count < num_workers)
        slurm: SLURM configuration dict (e.g., {"account": "data", "qos": "h100_lowest"})
        timeout: Maximum time in seconds to wait for all models to become RUNNING

    Returns:
        Dict with deployment status (cluster_id, app_names, statuses, deployment_time)
    """
    # Parse parameters if provided as JSON strings
    if isinstance(applications, str):
        applications = json.loads(applications)

    print(f"Starting deployment for cluster: {cluster_id}")
    print(f"Applications to deploy: {json.dumps(applications, indent=2)}")

    # Initialize CLI
    cli = matrix.Cli(cluster_id=cluster_id)

    # Check if cluster exists and has GPUs
    num_gpu = 0
    num_nodes = 0
    cluster_exists = False
    try:
        resources = cli.cluster.get_resources()
        num_gpu = resources["total_resources"].get("GPU", 0)
        # Count alive nodes (excluding head node)
        nodes = resources["nodes"]
        num_nodes = sum(
            1
            for node in nodes
            if node["Alive"] and not node["Resources"].get("node:__internal_head__")
        )
        cluster_exists = True
        print(f"Cluster exists with {num_nodes} worker nodes and {num_gpu} total GPUs")
    except Exception as e:
        print(f"Cluster does not exist or is not accessible: {e}")

    # Calculate workers needed
    workers_needed = max(0, num_workers - num_nodes)

    # Start cluster if needed
    if not cluster_exists:
        print(f"Starting cluster with {num_workers} workers...")
        cli.start_cluster(
            add_workers=num_workers,
            slurm=slurm,
        )
        print("Cluster started successfully")
    else:
        print("Reusing existing cluster")
        # Add more workers if needed
        if workers_needed > 0:
            print(
                f"Adding {workers_needed} workers to existing cluster (current: {num_nodes}, needed: {num_workers})..."
            )
            cli.start_cluster(add_workers=workers_needed, slurm=slurm)
        else:
            print(
                f"Cluster already has sufficient workers ({num_nodes} >= {num_workers})"
            )

    # Deploy all applications
    print("\nChecking existing application status...")
    app_names = []
    app_config_map = {}  # Map app_name to its configuration
    for app_config in applications:
        app_name = app_config.get("name")
        if not app_name:
            raise ValueError(
                f"Application configuration missing 'name' field: {app_config}"
            )
        app_names.append(app_name)
        app_config_map[app_name] = app_config

    # Check if all models are already running with requested replicas
    all_models_running = True
    to_remove = []
    for app_name in app_names:
        try:
            status, running_replicas = cli.app.app_status(app_name)
            requested_min_replica = app_config_map[app_name].get("min_replica", 1)

            print(
                f"  {app_name}: {status} "
                f"(running: {running_replicas}, requested min: {requested_min_replica})"
            )

            # Check both status and replica count
            if running_replicas < requested_min_replica:
                all_models_running = False
                if running_replicas == 0:  # no sign the previous deploy is working
                    to_remove.append(app_name)
        except Exception as e:
            print(f"  {app_name}: Not found or error checking status: {e}")
            all_models_running = False

    if all_models_running:
        print(
            f"\nAll {len(app_names)} model(s) are already RUNNING with required replicas. "
            "Skipping deployment."
        )
        return {
            "cluster_id": cluster_id,
            "app_names": app_names,
            "statuses": {app_name: "RUNNING" for app_name in app_names},
            "deployment_time": 0,
        }
    if to_remove:
        try:
            cli.deploy_applications(
                action=app_api.Action.REMOVE,
                applications=[{"name": name} for name in to_remove],
            )
        except Exception as e:
            print(f"Failed to remove apps {to_remove}: {e}")

    # Deploy all applications at once with REPLACE action
    print(f"\nDeploying {len(applications)} application(s)...")
    cli.deploy_applications(action=app_api.Action.REPLACE, applications=applications)

    # Wait for all applications to become RUNNING
    print("\nWaiting for all applications to become RUNNING...")
    start_time = time.time()
    all_running = False

    while not all_running:
        elapsed_time = time.time() - start_time

        # Check status of all apps
        statuses = {}
        for app_name in app_names:
            try:
                status, _ = cli.app.app_status(app_name)
                statuses[app_name] = status
            except Exception as e:
                statuses[app_name] = f"ERROR: {e}"

        if elapsed_time > timeout:
            print(f"\nTimeout reached ({timeout}s). Not all applications are RUNNING.")
            print("\nFinal status:")
            for app_name, status in statuses.items():
                print(f"  {app_name}: {status}")
            raise TimeoutError(
                f"Deployment timed out after {timeout}s. Not all applications reached RUNNING state."
            )

        # Print current status
        print(f"\n[{int(elapsed_time)}s elapsed] Current status:")
        for app_name, status in statuses.items():
            print(f"  {app_name}: {status}")

        # Check if all are RUNNING
        all_running = all(status == "RUNNING" for status in statuses.values())
        any_failed = any(status == "DEPLOY_FAILED" for status in statuses.values())
        if any_failed:
            raise RuntimeError(f"Deployment failed. {statuses}")

        if not all_running:
            time.sleep(10)  # Poll every 10 seconds

    # All applications are RUNNING
    print("\nAll applications are RUNNING!")

    total_time = time.time() - start_time
    print(f"\nDeployment completed successfully in {int(total_time)}s")

    return {
        "cluster_id": cluster_id,
        "app_names": app_names,
        "statuses": {app_name: "RUNNING" for app_name in app_names},
        "deployment_time": total_time,
    }


if __name__ == "__main__":
    fire.Fire(main)

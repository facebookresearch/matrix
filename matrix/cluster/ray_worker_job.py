# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import submitit

from matrix.common.cluster_info import ClusterInfo


@dataclass
class _RayWorkerConfiguration:
    """Configuration class for Ray worker job settings."""

    cluster_info: Any
    worker_wait_timeout_seconds: int = 60
    start_wait_time_seconds: int = 60
    environment: Dict[str, str] = field(default_factory=dict)
    logical_resources: str = "{}"

    def _determine_resource_allocation(self) -> tuple:
        """Dynamically determine CPU and GPU resources based on execution environment."""
        executor_type = os.environ.get("SUBMITIT_EXECUTOR", "slurm")

        if executor_type == "local":
            num_cpus = max((os.cpu_count() or 0) // 2, 1)
            num_gpus = len(
                [s for s in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if s]
            )
        else:
            num_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
            num_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", 0))

        return num_cpus, num_gpus


class _RayWorkerJobExecutor:
    """Executor class for managing Ray worker job initialization and execution."""

    @staticmethod
    def execute(config: _RayWorkerConfiguration) -> None:
        """
        Execute Ray worker job with provided configuration.

        Args:
            config (RayWorkerConfiguration): Configuration for Ray worker job
        """
        worker_env = os.environ.copy()
        worker_env.update(
            {
                "RAY_ADDRESS": f"{config.cluster_info.hostname}:{config.cluster_info.port}",
                "RAY_gcs_server_request_timeout_seconds": str(
                    config.worker_wait_timeout_seconds
                ),
                "RAY_raylet_start_wait_time_s": str(config.start_wait_time_seconds),
                **config.environment,
            }
        )

        num_cpus, num_gpus = config._determine_resource_allocation()

        print(worker_env)
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NULL')}")

        try:
            _RayWorkerJobExecutor._start_ray_worker(
                worker_env, num_cpus, num_gpus, config.logical_resources
            )
        finally:
            if os.path.exists(config.cluster_info.temp_dir):
                shutil.rmtree(config.cluster_info.temp_dir)

    @staticmethod
    def _start_ray_worker(
        worker_env: Dict[str, str],
        num_cpus: int,
        num_gpus: int,
        logical_resources: str,
    ) -> None:
        """
        Start Ray worker with specified environment and resources.

        Args:
            worker_env (Dict[str, str]): Worker environment variables
            num_cpus (int): Number of CPUs
            num_gpus (int): Number of GPUs
        """
        subprocess.run(
            [
                "ray",
                "start",
                "--address",
                "auto",
                "--block",
                "--num-cpus",
                str(num_cpus),
                "--num-gpus",
                str(num_gpus),
                "--resources",
                logical_resources,
            ],
            env=worker_env,
        )


# Example usage with added comments
class RayWorkerJob:
    def __init__(
        self,
        cluster_info: ClusterInfo,
        worker_wait_timeout_seconds: int,
        start_wait_time_seconds: int,  # TODO pass this around properly
        logical_resources: dict[str, Any],
    ):
        # Store the cluster information for later use
        self.cluster_info = cluster_info

        # Timeout for waiting for worker to become available
        self.worker_wait_timeout_seconds = 60

        # Initial wait time for Ray worker to start
        self.start_wait_time_seconds = 60
        self.logical_resources = logical_resources

    def __call__(
        self,
    ):
        """
        Execute the Ray worker job when the instance is called.

        This method:
        1. Creates a configuration for the Ray worker
        2. Executes the worker job using RayWorkerJobExecutor
        """
        config = _RayWorkerConfiguration(
            cluster_info=self.cluster_info,
            worker_wait_timeout_seconds=self.worker_wait_timeout_seconds,
            start_wait_time_seconds=self.start_wait_time_seconds,
            logical_resources=json.dumps(self.logical_resources),
        )
        _RayWorkerJobExecutor.execute(config)

    def checkpoint(
        self,
    ) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(self)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import typing as tp
from pathlib import Path

from matrix.common import JOB_MANAGER_STORE
from matrix.common.cluster_info import ClusterInfo
from matrix.utils.basics import convert_to_json_compatible
from matrix.utils.os import (
    create_symlinks,
    is_port_open,
    kill_proc_tree,
    run_subprocess,
)
from matrix.utils.ray import ACTOR_NAME_SPACE, get_ray_head_node, init_ray_if_necessary

_SLURM_KEY_ALIASES: dict[str, str] = {
    "slurm_account": "account",
    "slurm_qos": "qos",
    "slurm_partition": "partition",
}


def _normalize_slurm_keys(
    config: tp.Dict[str, tp.Union[str, int]],
) -> tp.Dict[str, tp.Union[str, int]]:
    """Map alternative slurm_* keys to their canonical form."""
    normalized = {}
    for key, value in config.items():
        normalized[_SLURM_KEY_ALIASES.get(key, key)] = value
    return normalized


def _get_slurm_default_requirements(requirements: dict[str, tp.Any]):
    """
    Extract SLURM partition info including CPU and memory requirements.

    Returns:
        dict: Requirements dictionary with partition, CPU, and memory info
    """
    try:
        partition = requirements.get("partition")
        if partition is None:
            # Get default partition
            output = subprocess.check_output(
                ["sinfo", "-h", "-o", "%P"], stderr=subprocess.PIPE
            )
            default_partition = [
                line.split("*")[0]
                for line in output.decode().splitlines()
                if "*" in line
            ]
            assert len(default_partition) == 1, f"Add partition to --slurm"
            partition = default_partition[0]
            requirements["partition"] = partition

        # Get detailed info for the partition
        # %P=partition, %c=CPUs, %m=memory(MB), %l=time_limit, %N=nodes
        sinfo_output = (
            subprocess.check_output(
                ["sinfo", "-h", "-p", partition, "-o", "%G %c %m"],
                stderr=subprocess.PIPE,
            )
            .decode()
            .strip()
        )

        if sinfo_output:
            lines = sinfo_output.splitlines()
            max_cpus = 0
            max_memory = 0
            max_gpus = 0
            gpu_type = None

            for line in lines:
                parts = line.split()
                if len(parts) >= 3:
                    gpu_info = _parse_gpu_gres(parts[0])
                    if gpu_info:
                        if gpu_info["count"] > max_gpus:
                            max_gpus = gpu_info["count"]
                            gpu_type = gpu_info["type"]

                    cpus = _parse_slurm_value(parts[1])
                    max_cpus = max(max_cpus, cpus)
                    memory = _parse_slurm_value(parts[2])
                    max_memory = max(max_memory, memory)

            requirements["cpus_per_task"] = max_cpus
            requirements["mem_gb"] = max_memory // 1024
            if max_gpus > 0:
                requirements["gpus_per_node"] = max_gpus

    except subprocess.CalledProcessError as e:
        print(f"Error running sinfo: {e}")
        raise
    except Exception as e:
        print(f"Error parsing sinfo output: {e}")
        raise

    return requirements


def _parse_slurm_value(value_str):
    """Parse SLURM numeric values (handles ranges and suffixes)."""
    value_str = value_str.rstrip("+")
    if "-" in value_str:
        return int(value_str.split("-")[-1])
    try:
        return int(value_str)
    except ValueError:
        numbers = re.findall(r"\d+", value_str)
        return int(numbers[-1]) if numbers else 0


def _parse_gpu_gres(gres_str):
    """Parse GPU GRES string: gpu:TYPE:COUNT(...)"""
    if not gres_str or gres_str == "(null)" or not gres_str.startswith("gpu:"):
        return None
    match = re.match(r"gpu:([^:]+):(\d+)", gres_str)
    if match:
        return {"type": match.group(1), "count": int(match.group(2))}
    return None


class RayCluster:
    """
    Manages the lifecycle of a Ray cluster on a Slurm-based system.

    This class provides functionalities to start, manage, and shutdown a Ray cluster,
    including the head node and worker nodes. It leverages Slurm for job scheduling
    and resource management.

    Args:
        cluster_id (str): A unique identifier for this Ray cluster.
        matrix_dir (Path): Base directory to store cluster-related information.
                            Defaults to ~/.matrix if not specified. This directory is
                            used for rendezvous data and logs, and can be used to
                            reconnect to an existing cluster.
    """

    def __init__(
        self,
        cluster_id: str,
        matrix_dir: Path,
    ):
        """Initializes a RayCluster instance."""
        self.cluster_id = cluster_id
        self.matrix_dir = matrix_dir
        print(f"cluster {self.cluster_id}")
        self.create_directory()
        print(f"logging to {self._log_dir.resolve()}")

        print(f"logging to {self._log_dir.resolve()}")

    def create_directory(self):
        """
        Creates the directory structure for the Ray cluster.

        This method ensures that the necessary directories for storing cluster data
        and logs are created.
        """
        self._cluster_dir.mkdir(parents=True, exist_ok=True)
        (self._cluster_dir / "jobs").mkdir(parents=True, exist_ok=True)

    @property
    def _cluster_dir(self) -> Path:
        """Returns the directory dedicated to this specific cluster's data."""
        return self.matrix_dir / self.cluster_id

    @property
    def _log_dir(self) -> Path:
        """Returns the directory where logs for this cluster are stored."""
        return self.matrix_dir / "logs" / self.cluster_id

    @property
    def _cluster_json(self) -> Path:
        """Returns the path to the JSON file storing cluster head information."""
        return self._cluster_dir / "head.json"

    def is_head_ready(self) -> bool:
        """Checks if the Ray head node has successfully started and cluster info is available."""
        return self._cluster_json.exists()

    def cluster_info(self) -> tp.Optional[ClusterInfo]:
        """
        Loads and returns the ClusterInfo object from the cluster's JSON file.

        Returns:
            Optional[ClusterInfo]:  Cluster information if the head node is ready,
                                    otherwise returns None.
        """
        try:
            with self._cluster_json.open("r") as f:
                return ClusterInfo(**json.load(f))
        except Exception as ex:
            print(f"failed to load head info: {ex}. Maybe it's not ready yet?")
            return None

    def _add_job(self, job):
        """
        Records a submitted Slurm job's ID to the cluster's job list.

        Args:
            job (submitit.Job): The submitted Slurm job object.
        """
        with (self._cluster_dir / "jobs" / f"{job.job_id}.json").open("w") as f:
            json.dump(
                {
                    "job_id": job.job_id,
                },
                fp=f,
            )

    def start_grafana(self, force: bool):
        """Start Prometheus and Grafana dashboard."""
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        from matrix.cluster.ray_dashboard_job import RayDashboardJob

        cluster_info = self.cluster_info()
        assert cluster_info is not None, "Head is not ready"
        init_ray_if_necessary(cluster_info)
        try:
            actor = ray.get_actor(RayDashboardJob.NAME, ACTOR_NAME_SPACE)
        except ValueError:  # Raised when actor does not exist
            actor = None
        if actor and force:
            try:
                ray.get(actor.cleanup.remote())
                ray.kill(actor)
            except:
                pass
            actor = None
        if actor:
            return "Grafana is already started"
        else:
            head_node = get_ray_head_node()
            actor = RayDashboardJob.options(  # type: ignore[attr-defined]
                name=RayDashboardJob.NAME,
                namespace=ACTOR_NAME_SPACE,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=head_node["NodeID"],
                    soft=False,
                ),
                lifetime="detached",
                num_cpus=0,
                num_gpus=0,
                max_restarts=3,  # Allow 3 automatic retries
                max_task_retries=-1,
            ).remote(
                cluster_info.temp_dir,
                cluster_info.prometheus_port,
                cluster_info.grafana_port,
            )
            ray.get(actor.start.remote())
            return "Successfully started Grafana dashboard"

    def start(
        self,
        add_workers: int,
        slurm: tp.Dict[str, tp.Union[str, int]] | None,
        local: tp.Dict[str, tp.Union[str, int]] | None,
        enable_grafana: bool = False,
        force_new_head: bool = False,
    ):
        """
        Starts a Ray cluster on Slurm.

        This method can either start a new cluster head node if one doesn't exist,
        or add worker nodes to an existing cluster.

        Args:
            add_workers (int): The number of worker nodes to start.
            slurm (dict, optional): resources requirements for slurm cluster.
                                    e.g., {'qos': '...', 'partition': '...', 'gpus-per-node': 8}.
            local (dict, optional): resources requirements for local cluster.
            slurm keys may also use the prefix ``slurm_`` (e.g., ``slurm_account``)
            which will be normalized to the canonical form (``account``).
            enable_grafana (bool): Whether to start Prometheus and Grafana
                                          for monitoring (default: True).
            force_new_head (bool): force to remove head.json if haven't run 'matrix stop_cluster'.
        """
        import submitit

        from matrix.cluster.ray_head_job import RayHeadJob
        from matrix.cluster.ray_worker_job import RayWorkerJob

        status: tp.Dict[str, tp.Any] = {}
        common_params = {"account", "partition", "qos", "exclusive", "timeout_min"}
        start_wait_time_seconds = 60
        worker_wait_timeout_seconds = 60
        requirements = slurm or local or {}
        requirements = _normalize_slurm_keys(requirements)
        executor = "slurm" if slurm else "local"

        if self._cluster_json.exists():
            cluster = self.cluster_info()
            assert cluster is not None, "Head is not ready"
            if not is_port_open(cluster.hostname, cluster.port):
                print(
                    f"Head node {cluster.hostname}:{cluster.port} is not reachable, stopping cluster first"
                )
                self.stop()
                self.create_directory()

        if force_new_head:
            # remove existing head.json
            if self._cluster_json.exists():
                self._cluster_json.unlink()

        if self._cluster_json.exists():
            print(f"Adding workers to existing cluster:\n{self.cluster_info()}")
            # todo: check the cluser is alive
        else:
            # start the head node
            s_executor = submitit.AutoExecutor(
                folder=str(self._log_dir),
                cluster=executor,
            )
            head_default_params = {"timeout_min": 10080, "cpus_per_task": 20}
            if add_workers == 0:
                head_params = requirements
            else:
                head_params = {
                    k: v for k, v in requirements.items() if k in common_params
                }
            head_params.update(
                {
                    key: value
                    for key, value in head_default_params.items()
                    if key not in head_params
                }
            )
            s_executor.update_parameters(
                name=f"ray_head_{self.cluster_id}",
                **head_params,
            )
            head_job = s_executor.submit(
                RayHeadJob.start_ray_head,
                self.cluster_id,
                self._cluster_json,
                worker_wait_timeout_seconds,
                executor,
            )
            self._add_job(head_job)
            create_symlinks(self._log_dir, "head", head_job.paths)
            print("head slurm job id:", head_job.job_id)

            job_submit_time = time.time()
            head_start_timeout = False
            while head_job.state != "RUNNING":
                print(f"Job {head_job.job_id} is in state: {head_job.state}")
                time.sleep(5)

                if time.time() - job_submit_time > start_wait_time_seconds:
                    if self.is_head_ready():
                        head_start_timeout = True
                        break

            if head_start_timeout:
                print(f"head may not have started, check manually!")
            else:
                print(f"Job {head_job.job_id} has started.")

            while not self.is_head_ready():
                print("Wait for head ready")
                time.sleep(5)
            cluster_info = self.cluster_info()
            cluster_info_dict = convert_to_json_compatible(cluster_info)
            print(
                f"ssh to head node:\nssh -L {cluster_info.dashboard_port}:localhost:{cluster_info.dashboard_port} -L {cluster_info.prometheus_port}:localhost:{cluster_info.prometheus_port} -L {cluster_info.grafana_port}:localhost:{cluster_info.grafana_port} {cluster_info.hostname}"  # type: ignore[union-attr]
            )
            print("\nHead Info:")
            print(json.dumps(cluster_info_dict, indent=2))

        if enable_grafana:
            self.start_grafana(force=True)

        # start the workers
        if add_workers > 0:
            s_executor = submitit.AutoExecutor(
                folder=str(self._log_dir), cluster=executor
            )
            default_params: dict[str, tp.Any] = {
                "ntasks_per_node": 1,
                "timeout_min": 10080,
            }
            partition = requirements.get("partition")
            if partition:
                default_params["partition"] = partition
            if executor == "slurm":
                default_params = _get_slurm_default_requirements(default_params)
            requirements.update(
                {
                    key: value
                    for key, value in default_params.items()
                    if key not in requirements
                }
            )
            print(requirements)
            s_executor.update_parameters(
                name=f"ray_worker_{self.cluster_id}",
                **requirements,
            )

            cluster_info = self.cluster_info()
            assert cluster_info is not None
            jobs = []
            with (
                s_executor.batch()
            ):  # TODO set slurm array max parallelism here, because we really want all jobs to be scheduled at the same time
                logical_resources = {
                    f"{key}-{value}": 1
                    for key, value in requirements.items()
                    if key in _SLURM_KEY_ALIASES.values()
                }

                for i in range(add_workers):
                    jobs.append(
                        s_executor.submit(
                            RayWorkerJob(
                                cluster_info,
                                worker_wait_timeout_seconds,
                                start_wait_time_seconds,
                                logical_resources,
                            )
                        )
                    )

            for j in jobs:
                create_symlinks(self._log_dir, f"worker", j.paths, True)
            print("workers slurm job ids:", [job.job_id for job in jobs])
            status["workers"] = [job.job_id for job in jobs]
            for j in jobs:
                self._add_job(j)

        status["cluster_info"] = self.cluster_info()
        return status

    def stop(self):
        """
        Shuts down the Ray cluster.
        """
        import ray

        cluster_info = self.cluster_info()
        assert cluster_info is not None, "Head is not ready"
        try:
            init_ray_if_necessary(cluster_info)
            ray.shutdown()
        except Exception as e:
            print(f"Ignore failures when shutdowning Ray: {e}")

        job_ids = [f.stem for f in (self._cluster_dir / "jobs").iterdir()]
        root_ids = list(set([i.split("_", maxsplit=2)[0] for i in job_ids]))
        run_subprocess(["scancel"] + root_ids)
        if cluster_info.executor == "local":
            for job in job_ids:
                try:
                    if job.isdigit():
                        kill_proc_tree(int(job), including_parent=False)
                except:
                    pass
            # clean local resources
            for pattern in [cluster_info.temp_dir, f":{cluster_info.port}"]:
                if pattern:
                    run_subprocess(["pkill", "-f", pattern, "-9"])
        for name in os.listdir(self._cluster_dir):
            if name == JOB_MANAGER_STORE:
                continue
            path = os.path.join(self._cluster_dir, name)
            (shutil.rmtree if os.path.isdir(path) else os.remove)(path)
        print(f"cluster {self.cluster_id} shutdown")

    def __enter__(self):
        # Start the cluster when entering the context
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        # Ensure the cluster is stopped when exiting the context
        self.stop()

    def get_resources(self):
        import ray

        cluster_info = self.cluster_info()
        assert cluster_info is not None, "Head is not ready"
        init_ray_if_necessary(cluster_info)
        return {
            "nodes": ray.nodes(),
            "total_resources": ray.cluster_resources(),
            "available_resources": ray.available_resources(),
        }

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

import ray

from matrix.common.cluster_info import ClusterInfo
from matrix.utils.os import find_free_ports, read_stdout_lines


class RayHeadJob:

    @staticmethod
    def start_ray_head(
        cluster_id: str,
        cluster_json_path: Path,
        worker_wait_timeout_seconds: int,
        executor: str,
    ):
        """Start the head node of the Ray cluster on slurm."""
        hostname = socket.gethostname()
        head_env = os.environ.copy()
        (
            port,
            client_server_port,
            dashboard_port,
            http_port,
            grpc_port,
            metrics_port,
            prometheus_port,
            grafana_port,
            sglang_dist_init_port,
            sglang_http_port,
            dashboard_agent_listen_port,
        ) = find_free_ports(11)
        # Configure environment variables
        head_env.update(
            {
                "RAY_ADDRESS": f"{hostname}:{port}",
                "RAY_gcs_server_request_timeout_seconds": str(
                    worker_wait_timeout_seconds
                ),
                "RAY_PROMETHEUS_HOST": f"http://localhost:{prometheus_port}",
                "RAY_GRAFANA_HOST": f"http://localhost:{grafana_port}",
            }
        )

        ip_address = socket.gethostbyname(hostname)
        print(f"Host {hostname}:{port}, IP {ip_address}")

        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            # Start Ray head process
            ray_process = subprocess.Popen(
                [
                    "ray",
                    "start",
                    "--head",
                    "--disable-usage-stats",
                    f"--port={port}",
                    f"--ray-client-server-port={client_server_port}",
                    f"--dashboard-port={dashboard_port}",
                    f"--metrics-export-port={metrics_port}",
                    f"--temp-dir={temp_dir}",
                    "--num-cpus",
                    "0",
                    "--num-gpus",
                    "0",
                    "--dashboard-host=0.0.0.0",
                    f"--dashboard-agent-listen-port={dashboard_agent_listen_port}",
                ],
                env=head_env,
                stdout=subprocess.PIPE,
                text=True,
            )

            # Verify Ray head start
            started = any(
                "ray start --address=" in line
                for line in read_stdout_lines(ray_process)
            )
            assert (
                started
            ), "Couldn't find head address in stdout. Check head.err for details"

            print(f"Head started, ip: {hostname}:{port} ({cluster_id})")
            info = ClusterInfo(
                hostname=hostname,
                port=int(port),
                client_server_port=int(client_server_port),
                dashboard_port=int(dashboard_port),
                metrics_port=int(metrics_port),
                http_port=int(http_port),
                grpc_port=int(grpc_port),
                prometheus_port=int(prometheus_port),
                grafana_port=int(grafana_port),
                sglang_dist_init_port=int(sglang_dist_init_port),
                sglang_http_port=int(sglang_http_port),
                dashboard_agent_listen_port=int(dashboard_agent_listen_port),
                temp_dir=temp_dir,
                executor=executor,
            )
            with cluster_json_path.open("w") as f:
                json.dump(dataclasses.asdict(info), f)

            while True:
                time.sleep(60)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from typing import List

import ray

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RayDashboardJob")


@ray.remote(max_restarts=10)
class RayDashboardJob:
    NAME = "system.ray_dashboard_job"
    """
    Ray Actor that manages Grafana and Prometheus subprocesses for monitoring.

    This actor runs and monitors two subprocesses (typically Grafana and Prometheus).
    If either subprocess terminates, the actor will also terminate, triggering Ray's
    automatic restart mechanism (up to max_restarts=10).

    The actor uses threading to monitor subprocess health and handle termination
    properly when the actor is killed.
    """

    def __init__(self, temp_dir: str, prometheus_port: int, grafana_port: int):
        """
        Initialize the RayDashboardJob actor.
        """
        self.head_env = os.environ.copy()
        self.temp_dir = temp_dir
        self.prometheus_port = prometheus_port
        self.grafana_port = grafana_port
        self.processes: List[subprocess.Popen[str]] = []
        self.monitor_thread: threading.Thread | None = None
        self.should_run = True
        self.pid = os.getpid()
        logger.info(f"RayDashboardJob initialized with PID {self.pid}")

    def start(self):
        """
        Start Grafana and Prometheus subprocesses and monitor their health.

        Returns:
            dict: Status information including actor PID and subprocess PIDs
        """
        # Start the subprocesses
        try:
            grafana_process = self._start_grafana()

            prometheus_process = self._start_prometheus()

            self.processes = [grafana_process, prometheus_process]

            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitor_processes)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

            return {
                "status": "started",
                "actor_pid": self.pid,
                "grafana_pid": grafana_process.pid,
                "prometheus_pid": prometheus_process.pid,
            }

        except Exception as e:
            logger.error(f"Error starting processes: {str(e)}")
            self.cleanup()
            raise e

    def _monitor_processes(self):
        """
        Monitor the health of subprocesses and terminate the actor if any subprocess stops.
        """
        logger.info("Process monitoring thread started")
        while self.should_run:
            for i, process in enumerate(self.processes):
                if process.poll() is not None:
                    process_name = "Grafana" if i == 0 else "Prometheus"
                    exit_code = process.poll()
                    logger.warning(
                        f"{process_name} process exited with code {exit_code}"
                    )

                    # Get process output for debugging
                    stdout, stderr = process.communicate()
                    logger.info(f"{process_name} stdout: {stdout}")
                    logger.error(f"{process_name} stderr: {stderr}")

                    # Clean up and terminate actor
                    logger.info("Terminating actor due to subprocess failure")
                    self.cleanup()
                    return

            # Check every second
            time.sleep(1)

    def cleanup(self):
        """
        Clean up resources by terminating all subprocesses.
        """
        logger.info("Running cleanup")
        self.should_run = False

        # Terminate all processes
        for i, process in enumerate(self.processes):
            if process.poll() is None:
                process_name = "Grafana" if i == 0 else "Prometheus"
                logger.info(f"Terminating {process_name} process (PID: {process.pid})")
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except Exception as e:
                    logger.error(f"Error terminating {process_name}: {str(e)}")

        # Wait for processes to terminate
        for i, process in enumerate(self.processes):
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process_name = "Grafana" if i == 0 else "Prometheus"
                logger.warning(
                    f"{process_name} didn't terminate gracefully, forcing kill"
                )
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except Exception:
                    pass

        logger.info("Terminating actor")
        ray.actor.exit_actor()

    def get_status(self):
        """
        Get the current status of the managed processes.

        Returns:
            dict: Status information for each process
        """
        status = {"actor_pid": self.pid, "processes": []}

        for i, process in enumerate(self.processes):
            process_name = "Grafana" if i == 0 else "Prometheus"
            process_status = (
                "running"
                if process.poll() is None
                else f"stopped (code: {process.poll()})"
            )
            status["processes"].append(  # type: ignore[attr-defined]
                {"name": process_name, "pid": process.pid, "status": process_status}
            )

        return status

    def _start_prometheus(self):
        head_env, temp_dir, port = self.head_env, self.temp_dir, self.prometheus_port

        prometheus_path = os.path.join(
            head_env["CONDA_PREFIX"], "prometheus/prometheus"
        )
        lock_file_path = f"{temp_dir}/session_latest/metrics/prometheus/data"
        if os.path.exists(lock_file_path):
            shutil.rmtree(lock_file_path)
            print("remove Prometheus data path")

        with (
            open(f"{temp_dir}/session_latest/logs/prometheus.out", "w") as stdout_file,
            open(f"{temp_dir}/session_latest/logs/prometheus.err", "w") as stderr_file,
        ):
            process = subprocess.Popen(
                [
                    prometheus_path,
                    "--config.file",
                    f"{temp_dir}/session_latest/metrics/prometheus/prometheus.yml",
                    "--storage.tsdb.path",
                    f"{temp_dir}/session_latest/metrics/prometheus/data/",
                    "--web.listen-address",
                    f"0.0.0.0:{port}",
                ],
                env=head_env,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                preexec_fn=os.setsid,
            )
        return process

    def _start_grafana(self):
        head_env, temp_dir, port = self.head_env, self.temp_dir, self.grafana_port
        grafana_path = os.path.join(
            head_env["CONDA_PREFIX"], "grafana/bin/grafana-server"
        )

        head_env["GF_SERVER_HTTP_PORT"] = str(port)
        head_env["GF_DATABASE_PATH"] = (
            f"{temp_dir}/session_latest/metrics/grafana/grafana.db"
        )
        conda_prefix = os.environ.get("CONDA_PREFIX")
        with (
            open(f"{temp_dir}/session_latest/logs/grafana.out", "w") as stdout_file,
            open(f"{temp_dir}/session_latest/logs/grafana.err", "w") as stderr_file,
        ):
            process = subprocess.Popen(
                [
                    grafana_path,
                    "--homepath",
                    f"{conda_prefix}/grafana/",
                    "--config",
                    f"{temp_dir}/session_latest/metrics/grafana/grafana.ini",
                    "web",
                ],
                env=head_env,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                preexec_fn=os.setsid,
            )

        return process

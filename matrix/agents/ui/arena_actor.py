# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import signal
import subprocess
import threading
import time

import ray

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("StreamlitArenaActor")


@ray.remote(max_restarts=3)
class StreamlitArenaActor:
    """Ray actor that runs the Arena Streamlit app as a managed subprocess.

    Follows the same pattern as RayDashboardJob: a detached actor pinned to the
    head node that launches a subprocess, monitors it, and exposes
    start / cleanup / get_status methods.
    """

    NAME = "system.arena"

    def __init__(self, port: int):
        self.port = port
        self.process: subprocess.Popen | None = None
        self.monitor_thread: threading.Thread | None = None
        self.should_run = True
        self.pid = os.getpid()
        logger.info("StreamlitArenaActor initialized (pid=%d, port=%d)", self.pid, port)

    def start(self):
        """Launch the Streamlit server and begin monitoring."""
        arena_script = os.path.join(os.path.dirname(__file__), "arena.py")
        if not os.path.exists(arena_script):
            raise FileNotFoundError(f"Arena script not found: {arena_script}")

        cmd = [
            "streamlit",
            "run",
            arena_script,
            f"--server.port={self.port}",
            "--server.headless=true",
            "--server.address=0.0.0.0",
            "--browser.gatherUsageStats=false",
        ]

        logger.info("Starting Streamlit: %s", " ".join(cmd))
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()

        return {
            "status": "started",
            "actor_pid": self.pid,
            "streamlit_pid": self.process.pid,
            "port": self.port,
        }

    def _monitor(self):
        """Watch the subprocess and terminate the actor if it dies."""
        logger.info("Monitor thread started")
        while self.should_run:
            if self.process and self.process.poll() is not None:
                exit_code = self.process.poll()
                logger.warning("Streamlit exited with code %s", exit_code)
                stdout, _ = self.process.communicate()
                if stdout:
                    logger.info("Streamlit output: %s", stdout[-2000:])
                self.cleanup()
                return
            time.sleep(2)

    def cleanup(self):
        """Terminate the Streamlit subprocess and exit the actor."""
        logger.info("Running cleanup")
        self.should_run = False
        if self.process and self.process.poll() is None:
            logger.info("Terminating Streamlit (pid=%d)", self.process.pid)
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except Exception as e:
                logger.error("Error terminating Streamlit: %s", e)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Force-killing Streamlit")
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except Exception:
                    pass
        logger.info("Terminating actor")
        ray.actor.exit_actor()

    def get_status(self):
        """Return current status of the Streamlit process."""
        if self.process is None:
            proc_status = "not started"
        elif self.process.poll() is None:
            proc_status = "running"
        else:
            proc_status = f"stopped (code: {self.process.poll()})"
        return {
            "actor_pid": self.pid,
            "port": self.port,
            "streamlit_pid": self.process.pid if self.process else None,
            "status": proc_status,
        }

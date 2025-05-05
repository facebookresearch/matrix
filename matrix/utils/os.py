# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import concurrent
import os
import select
import signal
import socket
import subprocess
import threading
import time
import traceback
import typing as tp
import uuid
from collections import deque
from contextlib import closing
from pathlib import Path

import portalocker
import psutil
import submitit


def kill_proc_tree(pid, including_parent=True):
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    print(children)
    for child in children:
        child.kill()
    gone, still_alive = psutil.wait_procs(children, timeout=5)
    if including_parent:
        parent.kill()
        parent.wait(5)


def find_free_ports(n):
    free_ports: set[int] = set()

    while len(free_ports) < n:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = s.getsockname()[1]
            free_ports.add(port)

    return list(free_ports)


def read_stdout_lines(proc: subprocess.Popen):
    """
    Yield lines from a subprocess's stdout without blocking.
    Args:
        proc (subprocess.Popen): The subprocess with stdout set to a pipe.
    Yields:
        str: Each line from the subprocess's stdout, stripped of whitespace.
    Raises:
        ValueError: If the subprocess's stdout is not a pipe.
    """
    if proc.stdout is None:
        raise ValueError(
            "Ensure stdout=subprocess.PIPE and text=True are set in Popen."
        )
    while True:
        ready_to_read, _, _ = select.select([proc.stdout], [], [], 0)
        if ready_to_read:
            output_line = proc.stdout.readline()
            if not output_line:
                break
            yield output_line.strip()


def create_symlinks(
    destination: Path,
    job_category: str,
    job_paths: submitit.core.utils.JobPaths,
    increment_index: bool = False,
):
    """Generate symbolic links for job's stdout and stderr in the specified directory with a formatted name."""

    def get_next_index(directory: Path, prefix: str) -> int:
        """Determine the next available index for symlink naming."""
        indices = {
            int(file.stem.split("_")[-1])
            for file in directory.glob(f"{prefix}_*.*")
            if file.suffix in {".err", ".out"}
        }
        return max(indices, default=-1) + 1

    def remove_existing_symlinks(directory: Path, prefix: str):
        """Remove existing symlinks if they exist."""
        for ext in (".err", ".out"):
            symlink = directory / f"{prefix}{ext}"
            if symlink.is_symlink():
                symlink.unlink()

    if increment_index:
        job_category = f"{job_category}_{get_next_index(destination, job_category)}"
    else:
        remove_existing_symlinks(destination, job_category)
    (destination / f"{job_category}.err").symlink_to(job_paths.stderr)
    (destination / f"{job_category}.out").symlink_to(job_paths.stdout)


def run_and_stream(logging_config, command, blocking=False):
    """Runs a subprocess, streams stdout/stderr in realtime, and ensures cleanup on termination."""
    remote = logging_config.get("remote", False)
    logger = logging_config["logger"]
    pid = None

    def log(str):
        if remote:
            logger.log.remote(f"[{pid}]" + str)
        else:
            logger.info(str)

    log(f"launch: {command}")

    """Runs a subprocess, streams stdout/stderr, and ensures cleanup."""
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,  # Run in a separate process group
    )
    pid = process.pid

    terminate_flag = threading.Event()
    stdout_buffer: tp.Deque[str] = deque(maxlen=10)

    def stream_output():
        """Reads and logs the subprocess output in real-time."""
        try:
            while not terminate_flag.is_set() and process.poll() is None:
                if process.stdout:
                    ready_to_read, _, _ = select.select([process.stdout], [], [], 0.1)
                    if ready_to_read:
                        line = process.stdout.readline()
                        if line:
                            log(line.strip())
                            stdout_buffer.append(line)
        except Exception as e:
            log(f"Error reading subprocess output: {e}")
        finally:
            # Make sure to read any remaining output
            if process.stdout:
                for line in process.stdout:
                    stdout_buffer.append(line)
                    log(line.strip())
                process.stdout.close()

    # Start log streaming in a separate thread to avoid blocking
    output_thread = threading.Thread(target=stream_output, daemon=True)
    output_thread.start()

    log(f"Launch proces {pid} with group {os.getpgid(pid)}")
    if not blocking:
        return process
    else:
        try:
            while True:
                exit_code = process.poll()
                if exit_code is not None:
                    print(f"Process finished with code {exit_code}")
                    break
                time.sleep(1)
            log(f"Process exited with code {exit_code}")
            stdout_content = "".join(stdout_buffer)
            return {
                "success": exit_code == 0,
                "exit_code": exit_code,
                "stdout": stdout_content,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "stdout": stdout_content,
            }
        finally:
            terminate_flag.set()
            output_thread.join(timeout=1.0)
            stop_process(process)
            log(f"Subprocess killed")


def stop_process(process):
    """Stops the subprocess and cleans up."""
    if process and process.poll() is None:
        print("Stopping subprocess...")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait()
        print("Subprocess stopped.")


def run_subprocess(command: tp.List[str]) -> bool:
    """
    Executes a command using subprocess.run and returns True if it runs successfully.
    Args:
        command (List[str]): The curl command to execute as a list of strings.
    Returns:
        bool: True if the command runs successfully, False otherwise.
    """
    print("Running command:", " ".join(command))
    try:
        # Execute the command
        result = subprocess.run(command, check=False, text=True)

        # Check the return code
        if result.returncode == 0:
            return True
        else:
            print(f"Command failed with return code {result.returncode}")
            return False
    except Exception as e:
        return False


def lock_file(filepath, mode, timeout=10, poll_interval=0.1):
    start_time = time.time()
    while True:
        try:
            return portalocker.Lock(
                filepath, mode, flags=portalocker.LockFlags.EXCLUSIVE
            )
        except portalocker.exceptions.AlreadyLocked:
            if (time.time() - start_time) >= timeout:
                raise TimeoutError(
                    f"Could not acquire lock for {filepath} within {timeout} seconds."
                )
            time.sleep(poll_interval)


def run_async(coro: tp.Awaitable[tp.Any]) -> tp.Any:
    """
    Run an async coroutine from a synchronous context.
    Handles cases where an event loop is already running (e.g., Jupyter, FastAPI).
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():

        def run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(run_in_new_loop).result()
    else:
        return loop.run_until_complete(coro)

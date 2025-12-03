# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import concurrent
import logging
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
from typing import Any, Callable, Dict, List, Optional

import portalocker
import psutil
from tqdm import tqdm

logger = logging.getLogger(__name__)


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


def is_port_available(port):
    """Check if a port is available on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return True
        except OSError:
            return False


def find_free_ports(n):
    free_ports: set[int] = set()

    while len(free_ports) < n:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = s.getsockname()[1]
            free_ports.add(port)

    return list(free_ports)


def is_port_open(host, port, timeout=2):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


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
    job_paths,
    increment_index: bool = False,
):
    """
    Generate symbolic links for job's stdout and stderr in the specified directory with a formatted name.

    Args:
        job_paths (submitit.core.utils.JobPaths): paths to the job's stdout and stderr files.
    """

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


def run_and_stream(
    logging_config,
    command,
    blocking=False,
    env=None,
    return_stdout_lines=10,
    skip_logging: str | None = None,
):
    """Runs a subprocess, streams stdout/stderr in realtime, and ensures cleanup on termination."""
    remote = logging_config.get("remote", False)
    logger = logging_config.get("logger")
    pid = None

    def log(str):
        if remote:
            logger.log.remote(f"[{pid}]" + str)
        elif logger is not None:
            logger.info(str)

    log(f"launch: {command}")
    if env is not None:
        extra_env = env
        env = os.environ.copy()
        env.update(extra_env)

    """Runs a subprocess, streams stdout/stderr, and ensures cleanup."""
    process = subprocess.Popen(
        command,
        shell=True if isinstance(command, str) else False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        preexec_fn=os.setsid,  # Run in a separate process group
    )
    pid = process.pid

    terminate_flag = threading.Event()
    stdout_buffer: tp.Deque[str] = deque(maxlen=return_stdout_lines)

    def stream_output():
        """Reads and logs the subprocess output in real-time."""
        try:
            while not terminate_flag.is_set() and process.poll() is None:
                if process.stdout:
                    ready_to_read, _, _ = select.select([process.stdout], [], [], 0.1)
                    if ready_to_read:
                        line = process.stdout.readline()
                        if line and (skip_logging is None or not skip_logging in line):
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

    try:
        pgid = os.getpgid(pid)
        group = str(pgid)
    except ProcessLookupError:
        group = "<terminated>"

    log(f"Launch process {pid} with group {group}")
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
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            print(f"Process group {process.pid} already terminated or does not exist")
            return
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
    return portalocker.Lock(
        filepath,
        mode,
        flags=portalocker.LockFlags.EXCLUSIVE,
        timeout=timeout,
        check_interval=poll_interval,
    )


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


def download_s3_dir(
    s3_path: str, cache_dir: str, dir_levels=1, exclude: str | None = None
):
    """
    Download contents of an S3 directory to a local cache directory.

    - s3_path: full S3 path to the directory (must end with slash or be treated as a directory)
    - cache_dir: local cache root
    - dir_levels: how many trailing components from the S3 path to include in the subdirectory
    """
    if not s3_path.endswith("/"):
        s3_path += "/"

    # Remove s3:// prefix
    if s3_path.startswith("s3://"):
        s3_path = s3_path[len("s3://") :]

    parts = s3_path.rstrip("/").split("/")
    subdir_name = os.path.join(*parts[-dir_levels:])
    dest_dir = os.path.join(cache_dir, subdir_name)
    os.makedirs(dest_dir, exist_ok=True)

    cmd = ["aws", "s3", "cp", f"s3://{s3_path}", dest_dir, "--recursive"]
    if exclude is not None:
        cmd.extend(["--exclude", exclude])
    print(cmd)
    downloaded = run_subprocess(cmd)
    return downloaded, dest_dir


async def batch_requests_async(
    func: Callable[..., Any],
    args_list: List[Dict[str, Any]],
    batch_size: int = 32,
    verbose=False,
) -> List[Any]:

    semaphore = asyncio.Semaphore(batch_size)
    results: List[Any] = [None] * len(args_list)

    async def worker(index: int, kwargs: Dict[str, Any]):
        async with semaphore:
            try:
                results[index] = await func(**kwargs)
            except Exception as e:
                logger.error(f"Error in batch request {index}: {e}", exc_info=True)
                results[index] = e

    tasks = [worker(i, kwargs) for i, kwargs in enumerate(args_list)]

    # Progress bar
    pbar = tqdm(
        total=len(tasks),
        desc=f"Batch {func.__name__ if hasattr(func, '__name__') else 'func'}",
        disable=not verbose,
    )
    for coro in asyncio.as_completed(tasks):
        await coro
        pbar.update(1)
    pbar.close()

    return results

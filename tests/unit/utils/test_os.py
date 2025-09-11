# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import multiprocessing
import os
import signal
import socket
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from matrix.utils.os import (
    batch_requests_async,
    create_symlinks,
    download_s3_dir,
    find_free_ports,
    is_port_available,
    is_port_open,
    kill_proc_tree,
    lock_file,
    read_stdout_lines,
    run_and_stream,
    run_async,
    run_subprocess,
    stop_process,
)


def _hold(path, evt, duration: float) -> None:
    with lock_file(path, "w"):
        evt.set()
        time.sleep(duration)


def test_kill_proc_tree():
    # Mocking psutil.Process and its methods
    with patch("psutil.Process") as MockProcess:
        mock_process = MockProcess.return_value
        mock_process.children.return_value = []
        mock_process.kill.return_value = None
        mock_process.wait.return_value = None

        # Call the function
        kill_proc_tree(1234)

        # Assertions
        mock_process.children.assert_called_once_with(recursive=True)
        mock_process.kill.assert_called_once()
        mock_process.wait.assert_called_once_with(5)


def test_find_free_ports():
    ports = find_free_ports(3)
    assert len(ports) == 3
    assert len(set(ports)) == 3  # Ensure all ports are unique


def test_run_and_stream():
    logger = Mock()
    command = "echo 'Hello, World!'"

    process = run_and_stream({"logger": logger}, command)

    assert process is not None
    assert isinstance(process, subprocess.Popen)


def test_stop_process():
    with (
        patch("os.killpg") as mock_killpg,
        patch("os.getpgid", return_value=1234) as mock_getpgid,
        patch("subprocess.Popen") as MockPopen,
    ):

        mock_process = MockPopen.return_value
        mock_process.poll.return_value = None
        mock_process.pid = 1234

        stop_process(mock_process)

        # Verify that os.getpgid was called with the correct PID
        mock_getpgid.assert_called_once_with(1234)

        # Verify that os.killpg was called with the correct process group ID and signal
        mock_killpg.assert_called_once_with(1234, signal.SIGTERM)


def test_run_and_stream_handles_process_lookup_error():
    """run_and_stream should handle ProcessLookupError when logging."""
    logger = Mock()
    with patch("os.getpgid", side_effect=ProcessLookupError):
        process = run_and_stream({"logger": logger}, "echo hi")
        assert process is not None
        assert isinstance(process, subprocess.Popen)


def test_is_port_checks(unused_tcp_port):
    port = unused_tcp_port
    assert is_port_available(port)
    sock = socket.socket()
    sock.bind(("localhost", port))
    sock.listen(1)
    try:
        assert not is_port_available(port)
        assert is_port_open("localhost", port)
    finally:
        sock.close()
    assert not is_port_open("localhost", port)


def test_read_stdout_lines():
    proc = subprocess.Popen(
        ["bash", "-c", "printf 'a\\nb\\n'"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert list(read_stdout_lines(proc)) == ["a", "b"]
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()

    proc2 = subprocess.Popen(["bash", "-c", "echo hi"])
    try:
        with pytest.raises(ValueError):
            next(read_stdout_lines(proc2))
    finally:
        proc2.wait()


def test_create_symlinks(tmp_path):
    dest = tmp_path / "links"
    dest.mkdir()
    out1 = tmp_path / "o1.txt"
    err1 = tmp_path / "e1.txt"
    out1.write_text("o1")
    err1.write_text("e1")
    jp1 = SimpleNamespace(stdout=out1, stderr=err1)

    create_symlinks(dest, "job", jp1)
    assert (dest / "job.out").resolve() == out1

    out2 = tmp_path / "o2.txt"
    err2 = tmp_path / "e2.txt"
    out2.write_text("o2")
    err2.write_text("e2")
    jp2 = SimpleNamespace(stdout=out2, stderr=err2)

    create_symlinks(dest, "job", jp2)
    assert (dest / "job.out").resolve() == out2

    create_symlinks(dest, "task", jp1, increment_index=True)
    create_symlinks(dest, "task", jp2, increment_index=True)
    assert (dest / "task_0.out").resolve() == out1
    assert (dest / "task_1.out").resolve() == out2


def test_run_and_stream_blocking_env_and_skip_logging():
    logger = Mock()
    cmd = ["bash", "-c", "echo $FOO; echo skip"]
    result = run_and_stream(
        {"logger": logger},
        cmd,
        blocking=True,
        env={"FOO": "BAR"},
        skip_logging="skip",
    )
    assert result["success"] is True
    assert result["exit_code"] == 0
    assert result["stdout"].splitlines()[0] == "BAR"


def test_run_subprocess():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = SimpleNamespace(returncode=0)
        assert run_subprocess(["echo", "hi"]) is True
        mock_run.assert_called_once()

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = SimpleNamespace(returncode=1)
        assert run_subprocess(["echo", "hi"]) is False


def test_lock_file_timeout_and_retry(tmp_path):
    path = tmp_path / "lock.txt"

    ctx = multiprocessing.get_context("spawn")

    # Acquire in background process to test timeout
    evt1 = ctx.Event()
    p1 = ctx.Process(target=_hold, args=(path, evt1, 0.2))
    p1.start()
    assert evt1.wait(5)
    with pytest.raises(TimeoutError):
        with lock_file(path, "w", timeout=0.1, poll_interval=0.02):
            pass
    p1.join()
    p1.close()

    # Acquire again with enough timeout to succeed
    evt2 = ctx.Event()
    p2 = ctx.Process(target=_hold, args=(path, evt2, 0.2))
    p2.start()
    assert evt2.wait(5)
    with lock_file(path, "w", timeout=1, poll_interval=0.02):
        pass
    p2.join()
    p2.close()


def test_download_s3_dir_builds_command(tmp_path):
    with patch("matrix.utils.os.run_subprocess", return_value=True) as mock_run:
        downloaded, dest = download_s3_dir("s3://bucket/path", str(tmp_path))

    expected_dest = tmp_path / "path"
    assert downloaded is True
    assert dest == str(expected_dest)
    mock_run.assert_called_once_with(
        ["aws", "s3", "cp", "s3://bucket/path/", str(expected_dest), "--recursive"]
    )


def test_download_s3_dir_with_exclude(tmp_path):
    with patch("matrix.utils.os.run_subprocess", return_value=False) as mock_run:
        downloaded, dest = download_s3_dir(
            "s3://bucket/dir", str(tmp_path), exclude="*.tmp"
        )

    expected_dest = tmp_path / "dir"
    assert downloaded is False
    assert dest == str(expected_dest)
    mock_run.assert_called_once_with(
        [
            "aws",
            "s3",
            "cp",
            "s3://bucket/dir/",
            str(expected_dest),
            "--recursive",
            "--exclude",
            "*.tmp",
        ]
    )


def test_run_async_sync_context():
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)

        async def sample():
            return 5

        assert run_async(sample()) == 5
    finally:
        loop.close()
        asyncio.set_event_loop(None)


@pytest.mark.asyncio
async def test_run_async_with_running_loop():
    async def sample():
        await asyncio.sleep(0.01)
        return "ok"

    assert run_async(sample()) == "ok"


@pytest.mark.asyncio
async def test_batch_requests_async():
    async def func(value: int, fail: bool = False):
        await asyncio.sleep(0.01)
        if fail:
            raise ValueError("bad")
        return value * 2

    args_list = [
        {"value": 1},
        {"value": 2},
        {"value": 3, "fail": True},
    ]

    results = await batch_requests_async(func, args_list, batch_size=2)
    assert results[:2] == [2, 4]
    assert isinstance(results[2], Exception)

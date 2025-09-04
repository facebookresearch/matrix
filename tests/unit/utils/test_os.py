# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from matrix.utils.os import (
    batch_requests_async,
    create_symlinks,
    find_free_ports,
    is_port_available,
    is_port_open,
    kill_proc_tree,
    read_stdout_lines,
    run_and_stream,
    run_async,
    run_subprocess,
    stop_process,
)


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


def test_run_async_sync_context():
    async def sample():
        return 5

    assert run_async(sample()) == 5


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


if __name__ == "__main__":
    pytest.main()

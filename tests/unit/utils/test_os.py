# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from matrix.utils.os import (
    find_free_ports,
    kill_proc_tree,
    run_and_stream,
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

if __name__ == "__main__":
    pytest.main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from types import ModuleType
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

# provide a minimal ray stub before importing job_manager
ray_stub = cast(Any, ModuleType("ray"))


def remote(*_args, **_kwargs):
    def decorator(obj):
        return obj

    return decorator


ray_stub.remote = remote
ray_stub.get_actor = lambda *args, **kwargs: None
ray_stub.wait = lambda *args, **kwargs: ([], [])
ray_stub.get = lambda obj: obj
ray_stub.cancel = lambda *args, **kwargs: None
sys.modules.setdefault("ray", ray_stub)

from matrix.job import job_manager


def dummy_deploy(apps):
    return True


def test_execute_task_sequence_logs_traceback():
    actor = Mock()
    actor.log.remote = Mock()

    def user_func():
        raise ValueError("boom")

    with patch("matrix.job.job_manager.ray.get_actor", return_value=actor):
        result = job_manager._execute_task_sequence(  # type: ignore[operator]
            user_func,
            [],
            {},
            [],
            dummy_deploy,
            None,
            None,
            1,
            "t1",
        )
    assert result["success"] is False
    assert result["step"] == "user_function"
    logs = "".join(call.args[0] for call in actor.log.remote.call_args_list)
    assert "ValueError" in logs
    assert "boom" in logs
    assert "Traceback" in logs

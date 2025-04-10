# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import time
import uuid
from typing import Any, Dict, Generator

import pytest
import ray

from matrix.cli import Cli


@pytest.fixture(scope="module")
def matrix_cluster() -> Generator[Any, Any, Any]:
    """Start and stop Ray for the duration of these tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cli = Cli(cluster_id=str(uuid.uuid4()), matrix_dir=temp_dir)
        cli.start_cluster(
            add_workers=1,
            slurm=None,
            local={"gpus_per_node": 0, "cpus_per_task": 2},
            enable_grafana=False,
        )
        with cli.cluster:
            yield cli


def test_deploy_hello(matrix_cluster: Cli) -> None:
    """Test hello app"""
    cli = matrix_cluster
    cli.deploy_applications(applications=[{"name": "hello", "app_type": "hello"}])
    time.sleep(60)  # todo: avoid sleep in test
    assert cli.check_health("hello")

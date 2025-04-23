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

import matrix
from matrix.cli import Cli
from matrix.job.job_utils import echo


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
    task_definitions = [
        {
            "func": "matrix.job.job_utils.echo",
            "args": ["hello "],
        },
        {
            "func": echo,
            "args": ["world!"],
        },
    ]
    job_def = {
        "task_definitions": task_definitions,
    }
    job_id = cli.job.submit(job_def)
    for _ in range(10):
        status = cli.job.status(job_id)
        if status["status"] in ["COMPLETED", "FAILED"]:
            break
        time.sleep(5)
    results = cli.job.get_results(job_id)
    assert status["status"] == "COMPLETED", results
    outputs = "".join([r["output"] for r in results.values()])
    assert outputs == "hello world!"

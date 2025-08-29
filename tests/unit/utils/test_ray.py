# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from matrix.common.cluster_info import ClusterInfo
from matrix.utils import ray as ray_utils


def test_get_ray_addresses():
    info = ClusterInfo(hostname="host", client_server_port=10001, dashboard_port=8265)
    assert ray_utils.get_ray_address(info) == "ray://host:10001"
    assert ray_utils.get_ray_dashboard_address(info) == "http://host:8265"


def test_status_helpers():
    assert ray_utils.status_is_success("RUNNING")
    for status in ["DEPLOY_FAILED", "DELETING"]:
        assert ray_utils.status_is_failure(status)
    for status in ["NOT_STARTED", "DEPLOYING", "UNHEALTHY"]:
        assert ray_utils.status_is_pending(status)

    for fn in [
        ray_utils.status_is_success,
        ray_utils.status_is_failure,
        ray_utils.status_is_pending,
    ]:
        assert not fn("UNKNOWN")

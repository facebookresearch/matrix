# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import typing as tp


@dataclasses.dataclass
class ClusterInfo:
    """
    cluster information class
    """

    hostname: tp.Optional[str] = None
    port: tp.Optional[int] = None
    client_server_port: tp.Optional[int] = None
    dashboard_port: tp.Optional[int] = None
    metrics_port: tp.Optional[int] = None
    http_port: tp.Optional[int] = None
    grpc_port: tp.Optional[int] = None
    prometheus_port: tp.Optional[int] = None
    grafana_port: tp.Optional[int] = None
    sglang_dist_init_port: tp.Optional[int] = None
    sglang_http_port: tp.Optional[int] = None
    dashboard_agent_listen_port: tp.Optional[int] = None
    temp_dir: tp.Optional[str] = None
    executor: tp.Optional[str] = None


def get_head_http_host(cluster_info: ClusterInfo) -> str:
    assert cluster_info.hostname
    local_mode = cluster_info.executor == "local"
    return "localhost" if local_mode else cluster_info.hostname

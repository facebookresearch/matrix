# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import logging
import time

from matrix.common.cluster_info import ClusterInfo, get_head_http_host
from matrix.utils.http import fetch_url
from matrix.utils.ray import get_ray_dashboard_address

logger = logging.getLogger("endpoint_cache")


class EndpointCache:
    def __init__(
        self, cluster_info, app_name, endpoint_template, ttl=360, serve_app=True
    ):
        self.lock = asyncio.Lock()
        self.cluster_info = cluster_info
        self.timestamp = None
        self.ttl = ttl
        self.serve_app = serve_app
        self.app_name = app_name
        self.endpoint_template = endpoint_template
        self.ray_address = get_ray_dashboard_address(cluster_info)
        self.http_host = get_head_http_host(cluster_info)
        self.ips = set()

    async def __call__(self, force_update=False):
        if time.time() - (self.timestamp or 0) > self.ttl or force_update:
            async with self.lock:
                if time.time() - (self.timestamp or 0) > self.ttl or force_update:
                    try:
                        if self.serve_app:
                            self.timestamp = time.time()
                            status, content = await fetch_url(
                                self.ray_address + "/api/serve/applications/",
                                headers={"Accept": "application/json"},
                            )
                            if status is not None and status == 200:
                                ray_query_result = json.loads(content)
                                head_ip = ray_query_result["controller_info"]["node_ip"]
                                self.ips = set([y["node_ip"] for x, y in ray_query_result["proxies"].items() if y["status"] == "HEALTHY" and y["node_ip"] != head_ip])  # type: ignore[attr-defined]
                            else:
                                raise Exception(f"status: {status}, {content}")
                        else:
                            self.ips = {self.http_host}

                    except Exception as e:
                        logger.warning(f"Error fetching endpoints: {e}")

        return [self.endpoint_template.format(host=ip) for ip in self.ips if ip]

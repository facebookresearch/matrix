# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import os
import signal
import subprocess
import threading
import time
import traceback
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Dict, List, Optional, Union

import aiohttp
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang_router.launch_router import launch_router, parse_router_args

from matrix.common.cluster_info import ClusterInfo
from matrix.utils.http import fetch_url, post_url
from matrix.utils.os import run_and_stream, stop_process
from matrix.utils.ray import ACTOR_NAME_SPACE, get_matrix_actors, get_ray_head_node

logger = logging.getLogger("ray.sglang")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class State(Enum):
    INIT = "init"
    RUNNING = "running"
    FAILED = "failed"


class ActorType(Enum):
    ROUTER = "router"
    WORKER = "worker"


GPU_PER_HOST = 8
CPU_PER_HOST = 8


def kill_actor(handle):
    ray.get(handle.kill.remote())
    ray.kill(handle)


# $ python -m sglang_router.launch_router --port 40000 --worker-urls http://worker_url_1 http://worker_url_2
@ray.remote(num_cpus=0)
class SglangRouterActor:
    def __init__(self, info):
        self.info = info
        self.http_port = self.info["http_port"]
        self.replica_port = self.info["replica_port"]
        self.process = None
        self.is_shutdown = False
        self.running_replicas = []

    def is_running(self):
        """Checks if the subprocess is still running."""
        return self.process is not None and self.process.poll() is None

    def launch_router(self, replica_ips):
        worker_urls = " ".join(
            [f"http://{ip}:{self.replica_port}" for ip in replica_ips]
        )
        command = f"""
        python -m sglang_router.launch_router \
            --host=0.0.0.0 --port {self.http_port} --worker-startup-timeout-secs=36000 --worker-startup-check-interval=10 \
            --policy=round_robin --worker-urls {worker_urls}
        """
        self.process = run_and_stream({"logger": logger}, command)

    async def update_replicas(self, running, dead):
        async with aiohttp.ClientSession() as session:
            tasks = [
                post_url(
                    session,
                    f"http://localhost:{self.http_port}/add_worker?url=http://{ip}:{self.replica_port}",
                )
                for ip in running
            ]
            await asyncio.gather(*tasks)
            if self.info["state"] == State.INIT and tasks:
                logger.info(f"Become running with {running}")
                self.info["state"] = State.RUNNING
                if self.process is None:
                    self.launch_router(running)
            self.running_replicas = running.copy()

            tasks = [
                post_url(
                    session,
                    f"http://localhost:{self.http_port}/remove_worker?url=http://{ip}:{self.replica_port}",
                )
                for ip in dead
            ]
            await asyncio.gather(*tasks)

    async def update(self):
        if not self.is_running():
            logger.warning(f"Router process {self.process} not running")
            self.info["state"] = State.FAILED
        return self.info

    def kill(self):
        self.is_shutdown = True
        stop_process(self.process)

    def get_running_replicas(self):
        return self.running_replicas

    async def start(
        self,
        cluster_info: ClusterInfo,
        app: Dict[str, Union[str, int]],
    ):
        """use sglang to deploy, run the router in head, break into pipeline parallel replicas"""
        # todo: maintain the list of active actors, for cleanup. it was working when we have the list before.
        app_name = app["name"]

        # hack
        router_port = cluster_info.sglang_http_port or 0
        replica_port = cluster_info.sglang_http_port or 0
        dist_init_port = cluster_info.sglang_dist_init_port or 0

        tp = int(app["tensor-parallel-size"])
        pp = int(app["pipeline-parallel-size"])
        min_replica = int(app["min_replica"])
        dp_size = min(min_replica, max(1, GPU_PER_HOST // tp))
        num_replica = max(1, min_replica // dp_size)
        # num_host = min_replica * pp // dp_size

        async def monitor_replicas():
            while not self.is_shutdown:
                try:
                    # get list of alive actors
                    logger.info(f"start monitoring app {app_name}")
                    actors = get_matrix_actors(
                        cluster_info, prefix=f"{app_name}_replica"
                    )
                    handles = [
                        ray.get_actor(actor["name"], actor["ray_namespace"])
                        for actor in actors
                    ]
                    infos = ray.get([ac.update.remote() for ac in handles])
                    num_actors = len(actors)
                    logger.info(f"Active actors {infos}")

                    # kill actors that failed
                    killed_actors = []
                    for i in range(num_actors):
                        if infos[i]["state"] == State.FAILED:
                            infos[i]["state"] = State.FAILED
                            logger.info(f"kill failed actor {actors[i]}")
                            kill_actor(handles[i])
                            killed_actors.append(i)
                    if killed_actors:
                        # if the replica head got killed, also kill the others in the replica
                        for index in range(len(killed_actors)):
                            if infos[index].get("rank") == 0:
                                head_name = infos[index]["name"]
                                for child in range(num_actors):
                                    if (
                                        infos[child].get("head_name") == head_name
                                        and infos[child]["state"] != State.FAILED
                                    ):
                                        infos[child]["state"] = State.FAILED
                                        logger.info(
                                            f"kill failed actor {actors[child]}"
                                        )
                                        kill_actor(handles[child])
                                        killed_actors.append(child)

                    active_nodes = [
                        infos[i]["NodeID"]
                        for i in range(num_actors)
                        if i not in killed_actors
                    ]
                    active_actors = [
                        infos[i]["name"]
                        for i in range(num_actors)
                        if i not in killed_actors
                    ]

                    # compute the delta
                    workers = []
                    head_node = None
                    for node in ray.nodes():
                        if node["Alive"]:
                            if node["Resources"].get("node:__internal_head__"):
                                head_node = node
                            elif node["NodeID"] not in active_nodes:
                                workers.append(node)

                    # start actors
                    # prefer partial ones
                    new_actors = []
                    new_ips = []
                    replica_counter = Counter(
                        [
                            infos[i]["replica_index"]
                            for i in range(num_actors)
                            if i not in killed_actors and "replica_index" in infos[i]
                        ]
                    )
                    replica_indexes = sorted(
                        replica_counter.keys(), key=lambda x: -replica_counter[x]
                    )
                    replica_indexes += [
                        i for i in range(num_replica) if i not in replica_indexes
                    ]
                    cur_worker_index = -1
                    for replica_index in replica_indexes:
                        head_ip = None
                        for rank in range(pp):
                            actor_name = (
                                f"{app_name}_replica-{replica_index}_node-{rank}"
                            )
                            head_name = f"{app_name}_replica-{replica_index}_node-0"
                            if head_name in active_actors:
                                index = active_actors.index(head_name)
                                head_ip = infos[index]["NodeManagerAddress"]
                                continue
                            if (
                                not actor_name in active_actors
                                and cur_worker_index < len(workers) - 1
                            ):
                                cur_worker_index += 1
                                worker = workers[cur_worker_index]
                                head_ip = (
                                    worker["NodeManagerAddress"]
                                    if rank == 0
                                    else head_ip
                                )
                                assert head_ip
                                info = {
                                    "type": ActorType.WORKER,
                                    "state": State.INIT,
                                    "NodeID": worker["NodeID"],
                                    "NodeManagerAddress": worker["NodeManagerAddress"],
                                    "replica_index": replica_index,
                                    "rank": rank,
                                    "name": actor_name,
                                    "head_name": head_name,
                                    "head_ip": head_ip,
                                    "http_port": replica_port,
                                    "dist_init_port": dist_init_port,
                                }
                                actor1 = SglangActor.options(  # type: ignore[attr-defined]
                                    name=actor_name,
                                    namespace="matrix",
                                    lifetime="detached",  # Actor remains even if script exits
                                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                                        node_id=worker["NodeID"],
                                        soft=False,
                                    ),
                                    num_cpus=CPU_PER_HOST,
                                    num_gpus=GPU_PER_HOST,
                                    max_restarts=3,  # Allow 3 automatic retries
                                    max_task_retries=-1,
                                ).remote(
                                    app, info
                                )
                                new_actors.append(actor1)
                                new_ips.append(head_ip)

                    # compute the active replica
                    dead_replica, running_replica, init_replica = [], [], new_ips
                    for i in range(num_actors):
                        info = infos[i]
                        ip = infos[i]["NodeManagerAddress"]
                        if info["type"] == ActorType.WORKER and info["rank"] == 0:
                            if i in killed_actors:
                                dead_replica.append(ip)
                            elif info["state"] == State.RUNNING:
                                running_replica.append(ip)
                            elif info["state"] == State.INIT:
                                init_replica.append(ip)
                    await self.update_replicas(running_replica, dead_replica)
                    logger.info(
                        f"update replicas: running {running_replica}, init {init_replica}, dead_replica {dead_replica}"
                    )

                    # may timeout when an actor is PENDING_CREATION due to resources
                    ray.get(
                        [actor1.start.remote() for actor1 in new_actors], timeout=120
                    )
                    if new_actors:
                        await asyncio.sleep(30)
                    else:
                        await asyncio.sleep(10)
                except Exception as e:
                    error_message = traceback.format_exc()
                    logger.warning(f"Ignore exception {e} details {error_message}")
                    await asyncio.sleep(30)

        def run_async_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(monitor_replicas())

        # Run the async function in a separate thread
        thread = threading.Thread(target=run_async_in_thread, daemon=True)
        thread.start()


@ray.remote
class SglangActor:
    def __init__(self, app, info):
        self.app = app
        self.info = info
        self.process = None  # Store the subprocess reference

    def start(self):
        """Starts the subprocess and captures its output in real-time."""
        model_name = self.app["model_name"]
        tp = self.app["tensor-parallel-size"]
        dp = min(self.app["min_replica"], max(1, GPU_PER_HOST // tp))
        pp = self.app["pipeline-parallel-size"]
        nnodes = self.app["pipeline-parallel-size"]

        dist_init_addr = self.info["head_ip"] + ":" + str(self.info["dist_init_port"])
        rank = self.info["rank"]
        http_port = self.info["http_port"]
        command = f"""
        python -m sglang.launch_server --model-path {model_name} \
          --tp {tp*pp} --dp {dp} --nnodes {nnodes} --node-rank {rank} --trust-remote-code \
          --host 0.0.0.0 --port {http_port} --context-length {self.app["max-model-len"]} \
          --max-running-requests {self.app["max_ongoing_requests"]} --mem-fraction-static {self.app["gpu-memory-utilization"]} \
          --grammar-backend xgrammar
        """
        if nnodes > 1:
            command = command.rstrip() + f" --dist-init-addr {dist_init_addr}"

        self.process = run_and_stream({"logger": logger}, command)

    def is_running(self):
        """Checks if the subprocess is still running."""
        return self.process is not None and self.process.poll() is None

    async def update(self):
        """also check health and update state"""
        logger.info("calling update")
        if not self.is_running():
            self.info["state"] = State.FAILED
        else:
            head_ip = self.info["head_ip"]
            http_port = self.info["http_port"]
            status, content = await fetch_url(f"http://{head_ip}:{http_port}/health")
            logger.info(f"done calling update {status} {content}")
            state = self.info["state"]
            if status == 200:
                self.info["state"] = State.RUNNING
            elif state == State.RUNNING and status != 200:
                self.info["state"] = State.FAILED
        return self.info

    def kill(self):
        stop_process(self.process)


# @ray.remote
def deploy_app(
    log_dir: Path,
    cluster_info: ClusterInfo,
    app: Dict[str, Union[str, int]],
):
    """use sglang to deploy, run the router in head, break into pipeline parallel replicas"""
    # todo: maintain the list of active actors, for cleanup. it was working when we have the list before.
    app_name = app["name"]
    is_r1 = app.get("model_size") == "deepseek-r1"
    if is_r1:
        assert int(app["pipeline-parallel-size"]) in [
            1,
            2,
        ], "deepseek-r1 only support 8 or 16 GPUs"

    # hack
    router_port = cluster_info.sglang_http_port or 0
    replica_port = cluster_info.sglang_http_port or 0
    dist_init_port = cluster_info.sglang_dist_init_port or 0

    head_node = get_ray_head_node()
    router_name = f"{app_name}_router"
    # launch router
    info = {
        "type": ActorType.ROUTER,
        "state": State.INIT,
        "NodeID": head_node["NodeID"],
        "NodeManagerAddress": head_node["NodeManagerAddress"],
        "name": router_name,
        "http_port": router_port,
        "replica_port": replica_port,
    }

    router = SglangRouterActor.options(  # type: ignore[attr-defined]
        name=f"{app_name}_router",
        namespace=ACTOR_NAME_SPACE,
        lifetime="detached",
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=head_node["NodeID"],
            soft=False,
        ),
        num_cpus=0,
        num_gpus=0,
        max_restarts=3,  # Allow 3 automatic retries
        max_task_retries=-1,
    ).remote(info)
    ray.get(router.start.remote(cluster_info, app))

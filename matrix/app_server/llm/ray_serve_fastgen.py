# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import logging
import os
import socket
import subprocess as sp
import sys
import tempfile
from dataclasses import dataclass
from inspect import Parameter, signature
from multiprocessing.connection import Client, Connection, Listener
from queue import Queue
from threading import Thread
from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4

import ray
import torch
from fastapi import FastAPI
from fastgen.generate import Fastgen, GenArgs, Packet
from fastgen.utils.loading import HfLoader
from fastgen.utils.tokenizer import BaseTokenizer
from ray import serve
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

logger = logging.getLogger("ray.serve")

app = FastAPI()


@dataclass
class Request:
    messages: list[dict[str, str]]
    temperature: float = 0.8
    top_p: float = 0.95  # ignored
    max_tokens: int = 0
    min_tokens: int = 0  # ignored
    model: Optional[str] = None  # ignored
    n: int = 1
    stop: Optional[str] = None  # ignored
    seed: Optional[int] = None  # ignored
    logprobs: Optional[bool] = None  # ignored
    top_logprobs: Optional[int] = None  # ignored
    extra_headers: Optional[dict] = None  # ignored
    extra_body: Optional[dict] = None  # ignored


def worker_enqueuer(c: Connection, q: Queue) -> None:
    "Enqueuer thread."
    while True:
        msg, payload = c.recv()
        if msg == "quit":
            q.put(None)
            return

        if msg == "gen":
            rid, req, toks = payload
            if req.max_tokens <= 0:
                max_gen: Optional[int] = None
            else:
                max_gen = req.max_tokens
            for _ in range(req.n):
                q.put(
                    Packet(
                        thread_id=rid,
                        temperature=req.temperature,
                        max_gen=max_gen,
                        tokens=toks,
                    )
                )


def worker_main(
    args: argparse.Namespace,
    rdv_dir: str,
    rank: int,
    host_port: Tuple[str, int],
) -> None:
    world_size = args.tensor_parallel_size
    pid = os.getpid()
    device_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
    logger.warning(
        f"Worker {rank}, world_size {world_size}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, RDV_DIR={rdv_dir}"
    )

    c = Client(host_port)

    os.environ["ENABLE_INTRA_NODE_COMM"] = "1"
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

    device = torch.device(f"cuda:0")
    torch.cuda.set_device(device)
    if world_size == 1:
        mesh: Optional[DeviceMesh] = None
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=args.tensor_parallel_size,
            init_method=f"file://{rdv_dir}/rdv",
        )
        mesh = init_device_mesh("cuda", (world_size,))
    logger.warning(f"Done init_process_group")

    torch.manual_seed(777)
    torch.set_default_device(device)

    loader = HfLoader(args.model)
    fg = Fastgen.build(
        loader=loader,
        gen_args=GenArgs(),
        tp_mesh=mesh,
        device=device,
    )
    logger.warning(f"Done init Fastgen")
    c.send("ready")

    q: Queue = Queue()
    enqueuer_thread = Thread(target=worker_enqueuer, args=(c, q), daemon=True)
    enqueuer_thread.start()

    for pkt in fg.generate(q):
        if rank == 0:
            c.send((pkt.thread_id, pkt.tokens))

    enqueuer_thread.join()
    fg.destroy()
    if mesh is not None:
        torch.distributed.destroy_process_group()


class WorkerWrapper:

    def __init__(
        self,
        args: argparse.Namespace,
        rdv_dir: str,
        rank: int,
        host_port: Tuple[str, int],
    ):

        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)

        self.args = args
        self.rdv_dir = rdv_dir
        self.rank = rank

        self.process = mp.Process(
            target=worker_main,
            args=(
                args,
                rdv_dir,
                rank,
                host_port,
            ),
        )
        self.process.start()


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 8,
        "target_ongoing_requests": 64,
    },
    max_ongoing_requests=64,  # make this large so that multi-turn can route to the same replica
)
@serve.ingress(app)
class FastgenDeployment:

    def __init__(
        self,
        args: argparse.Namespace,
    ):
        logger.warning(f"Starting with engine args: {args} and env: {os.environ}")
        self.args = args
        self.workers: list[tuple[ActorHandle, Connection]] = []
        self.handles: dict[str, asyncio.Queue[list[int]]] = {}
        self.running: bool = True
        loader = HfLoader(args.model)
        self.tokenizer: BaseTokenizer = loader.load_tokenizer()

        self.setup(args)

    def receiver(self, loop: asyncio.AbstractEventLoop):
        """Receive from the rank 0 worker and put the response into the queue for that request"""
        while self.running:
            rid, tokens = self.workers[0][1].recv()

            # Find the asyncio.Queue handle
            hnd = self.handles.get(rid)
            if hnd is None:
                logger.warning(f"Got completion for unknown handle {rid}")
                continue

            # Schedule async put into the queue from this thread
            asyncio.run_coroutine_threadsafe(hnd.put(tokens), loop)  # type: ignore[func-returns-value]

    def setup(self, args):
        placement_group = ray.util.get_current_placement_group()
        bundle_indices = []
        for bundle_id, bundle in enumerate(placement_group.bundle_specs):
            if bundle.get("GPU", 0):
                bundle_indices.append(bundle_id)
        bundle_indices = bundle_indices[: args.tensor_parallel_size]

        hostname = socket.gethostname()
        listener = Listener((hostname, 0), "AF_INET", authkey=None)
        host, port = listener.address  # type: ignore[misc]
        logger.info(f"Listener started on host={host}, port={port}")

        with tempfile.TemporaryDirectory() as rdv_dir:
            for rank, bundle_id in enumerate(bundle_indices):

                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=bundle_id,
                )

                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=1,
                    scheduling_strategy=scheduling_strategy,
                )(WorkerWrapper).remote(self.args, rdv_dir, rank, (host, port))

                conn = listener.accept()
                logger.info(f"Connection accepted from: {conn}")

                self.workers.append((worker, conn))

            # wait for workers to be ready
            for _, c in self.workers:
                logger.info("Waiting for worker to be ready")
                _ = c.recv()

        loop = asyncio.get_event_loop()
        receiver_thread = Thread(target=self.receiver, args=(loop,), daemon=True)
        receiver_thread.start()

    @app.post("/v1/chat/completions")
    async def create_chat_completion(self, request: Request):
        """OpenAI-compatible HTTP endpoint."""
        logger.debug(f"Request: {request}")

        rq = request  # Request(**completion_request)
        rid = uuid4().hex
        prompt_tokens = self.tokenizer.encode_dialog(rq.messages)
        hnd: asyncio.Queue = asyncio.Queue()
        self.handles[rid] = hnd
        for _, c in self.workers:
            c.send(("gen", (rid, rq, prompt_tokens)))
        rsp = {
            "id": rid,
            "choices": [],
            "created": 0,
            "model": rq.model,
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 0,
                "prompt_tokens": len(prompt_tokens),
                "total_tokens": len(prompt_tokens),
            },
        }
        for ix in range(rq.n):
            tokens = await hnd.get()
            rsp["usage"]["completion_tokens"] += len(tokens)  # type: ignore[index]
            rsp["usage"]["total_tokens"] += len(tokens)  # type: ignore[index]
            if tokens[-1] in self.tokenizer.stop_tokens:
                tokens = tokens[:-1]
                stopped = True
            else:
                stopped = False
            rsp["choices"].append(  # type: ignore[attr-defined]
                {
                    "index": ix,
                    "message": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(tokens),
                    },
                    "finish_reason": "stop" if stopped else "length",
                }
            )
        del self.handles[rid]
        return rsp

    def __del__(self):
        self.running = False


def parse_args(cli_args: Dict[str, str]):
    """Parses engine args based on CLI inputs."""
    parser = argparse.ArgumentParser(description="Example with type annotations")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="model name",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Size of tensor parallelism (int)",
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=1,
        help="Size of pipeline parallelism (int)",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="Enable prefix caching (bool)",
    )
    parser.add_argument(
        "--max-ongoing-requests",
        type=int,
        default=100,
        help="Maximum number of ongoing requests (int)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Maximum model length (int)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (float)",
    )

    arg_strings = []
    for key, value in cli_args.items():
        if value is None:
            arg_strings.extend([f"--{key}"])
        else:
            arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments."""  # noqa: E501
    accelerator = "GPU"
    args = parse_args(cli_args)

    tp = args.tensor_parallel_size
    pp = args.pipeline_parallel_size
    assert pp == 1, "Pipeline parallelism > 1 is not supported in fastgen"
    logger.info(f"Tensor parallelism = {tp}, Pipeline parallelism = {pp}")
    pg_resources = []
    pg_resources.append({"CPU": 1})  # for the deployment replica
    for i in range(tp * pp):
        pg_resources.append({"CPU": 4, accelerator: 1})  # for the actors

    # We use the "STRICT_PACK" strategy below to ensure all actors are placed on
    # the same Ray node.
    return FastgenDeployment.options(  # type: ignore[attr-defined]
        placement_group_bundles=pg_resources,
        placement_group_strategy="STRICT_PACK" if pp == 1 else "PACK",
    ).bind(
        args,
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import logging
import os
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from fastapi import FastAPI
from ray import get_runtime_context, serve
from starlette.requests import Request
from starlette.responses import JSONResponse
from torch.utils.data import DataLoader, Dataset

from matrix.app_server.vision.utils import SamplingMode, TorchCodecVideoDataset

logger = logging.getLogger("ray.serve")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

TIMEOUT = 10

app = FastAPI()


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "target_ongoing_requests": 64,
    },
    ray_actor_options={"num_cpus": 4},
    max_ongoing_requests=100,
)
@serve.ingress(app)
class OpticalFlowDeployment:
    def __init__(
        self,
        torch_batch_size: int = 64,
        return_flow: bool = False,
        motion_score: bool = True,
    ):

        self.num_workers = self._find_max_num_workers()
        self.batch_size = torch_batch_size

        self.return_flow = return_flow
        self.calc_motion_score = motion_score

    def _find_max_num_workers(self):
        assigned_resources = get_runtime_context().get_assigned_resources()
        num_cpus_for_replica = int(assigned_resources.get("CPU", 1))
        num_workers = max(1, num_cpus_for_replica - 1)
        return num_workers

    def _calculate_motion_metrics(
        self,
        data_loader: DataLoader,
    ) -> Dict[str, float | List[List[float]]]:

        motion_scores: List[float] = []
        optical_flows: List[np.ndarray] = []

        prev_gray: np.ndarray | None = None

        # Iterate through batches from the DataLoader
        for batch in data_loader:
            # The dataloader yields batches of tensors, shape (Batch, C, H, W)
            video_batch_tensor = batch["frames"]

            # Iterate through each frame in the batch
            for i in range(video_batch_tensor.shape[0]):
                frame_tensor = video_batch_tensor[i]

                # --- Convert Tensor to OpenCV format ---
                # 1. Permute from (C, H, W) to (H, W, C)
                # 2. Convert to NumPy array
                # 3. OpenCV expects BGR, but since we convert to grayscale, RGB vs BGR doesn't matter.
                frame_numpy = frame_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
                gray = cv2.cvtColor(frame_numpy, cv2.COLOR_RGB2GRAY)

                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )

                    if self.return_flow:
                        optical_flows.append(flow)

                    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                    if self.calc_motion_score:
                        motion_scores.append(float(np.mean(magnitude)))

                prev_gray = gray

        scores: Dict[str, float | List[List[float]]] = {}

        if self.calc_motion_score:
            scores["motion_score"] = (
                float(np.mean(motion_scores)) if motion_scores else 0.0
            )

        if self.return_flow:
            scores["flow"] = [flow.tolist() for flow in optical_flows]

        return scores

    @app.post("/run")
    async def run(self, request: Request) -> JSONResponse:

        if request.method == "POST":
            request_json = await request.json()
            timeout = request_json.get("timeout", TIMEOUT)

        else:
            return JSONResponse(
                {"error": "Data must be in JSON format!"}, status_code=400
            )

        try:
            if "video" not in request_json:
                raise ValueError("Request must contain 'video' field with 'file_path'.")

            video_path = request_json["video"].get("file_path")
            if video_path is None:
                raise ValueError("Request must contain 'file_path' field in 'video'.")

            start: int = request_json["video"].get("start", 0)
            end: Optional[int] = request_json["video"].get("end", None)
            mode: SamplingMode = request_json["video"].get("mode", SamplingMode.FRAMES)
            sampling_rate: Optional[float] = request_json["video"].get(
                "sampling_rate", None
            )
            stride: Union[int, float] = request_json["video"].get("stride", 1)

            dataset = TorchCodecVideoDataset(
                video_path=video_path,
                start=start,
                end=end,
                mode=mode,
                stride=stride,
                sampling_rate=sampling_rate,
            )
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
            )

            result: Dict[str, float | List[List[float]]] = await asyncio.to_thread(
                self._calculate_motion_metrics,
                data_loader,
            )

            return JSONResponse(
                {
                    "success": True,
                    **result,
                },
                status_code=200,
            )

        except Exception as e:
            return JSONResponse(
                {
                    "success": False,
                    "error_msg": str(e),
                },
                status_code=500,
            )


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments."""

    argparse = ArgumentParser()
    argparse.add_argument(
        "--torch_batch_size",
        type=int,
        default=64,
        help="Batch size for torch data loading.",
    )
    argparse.add_argument(
        "--return_flow",
        type=bool,
        default=False,
        help="Whether to return optical flow data.",
    )
    argparse.add_argument(
        "--motion_score",
        type=bool,
        default=True,
        help="motion score calculation.",
    )

    arg_strings = []
    for key, value in cli_args.items():
        if value is not None:
            arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)

    args = argparse.parse_args(arg_strings)

    return OpticalFlowDeployment.bind(  # type: ignore[attr-defined]
        torch_batch_size=args.torch_batch_size,
        return_flow=args.return_flow,
        motion_score=args.motion_score,
    )

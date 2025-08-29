# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import logging
import os
import traceback
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Union

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
import torch
import torchvision.transforms.v2 as T
from fastapi import FastAPI
from PIL import Image
from ray import get_runtime_context, serve
from starlette.requests import Request
from starlette.responses import JSONResponse
from torch.utils.data import DataLoader, Dataset

from matrix.app_server.vision.utils import (
    SamplingMode,
    TorchCodecVideoDataset,
    execute_with_retry,
    get_image_transform,
)

logger = logging.getLogger("ray.serve")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

TIMEOUT = 10

app = FastAPI()


def _custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle batching of video tensors and metadata.
    """
    # Stack all the video tensors into a single batch tensor
    video_tensors = torch.stack([item["frames"] for item in batch], dim=0)

    # Keep the metadata as a simple list of dictionaries
    metadata_list = [item["meta"] for item in batch]

    return {
        "frames": video_tensors,
        "meta": metadata_list,  # This is now a list of dicts, not a dict of lists/tensors
    }


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "target_ongoing_requests": 64,
    },
    ray_actor_options={"num_cpus": 10, "num_gpus": 1},
    max_ongoing_requests=100,
)
@serve.ingress(app)
class PerceptionEncoderDeployment:
    def __init__(
        self,
        model_name: str,
        torch_batch_size: Optional[int] = None,
    ):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model = (
            pe.CLIP.from_config(self.model_name, pretrained=True).to(self.device).eval()
        )
        self.model = torch.compile(model, mode="reduce-overhead")
        self.preprocess = get_image_transform(self.model.image_size)

        self.num_workers = self._find_max_num_workers()
        self.batch_size = (
            self._find_optimal_batch_size()
            if torch_batch_size is None
            else torch_batch_size
        )

    def _find_max_num_workers(self):
        assigned_resources = get_runtime_context().get_assigned_resources()
        num_cpus_for_replica = int(assigned_resources.get("CPU", 1))
        num_workers = max(1, num_cpus_for_replica - 1)
        return num_workers

    def _find_optimal_batch_size(self, initial_batch_size=2048):
        """
        Performs a binary search to find the largest batch size that fits in GPU memory
        by creating a dummy input tensor based on the model's configuration.
        """
        image_size = self.model.image_size

        low = 1
        high = initial_batch_size
        optimal_batch_size = 0

        while low <= high:
            mid = (low + high) // 2
            if mid == 0:
                break

            try:
                dummy_batch = torch.zeros(
                    (mid, 3, image_size, image_size), device=self.device
                )

                with torch.no_grad():
                    self.model.encode_image(dummy_batch)

                optimal_batch_size = mid
                low = mid + 1

            except torch.cuda.OutOfMemoryError:
                high = mid - 1

            finally:
                # Clear cache after every attempt
                torch.cuda.empty_cache()

        return max(optimal_batch_size, 1)

    @app.post("/run")
    async def run(self, request: Request) -> JSONResponse:

        if request.method == "POST":
            request_json = await request.json()
            timeout = request_json.get("timeout", TIMEOUT)

        else:
            return JSONResponse(
                {"error": "Data must be in JSON format!"}, status_code=400
            )

        output = []
        try:
            if "video" in request_json:
                video_path = request_json["video"].get("file_path")
                if video_path is None:
                    return JSONResponse(
                        {"error": "Request must contain 'file_path' field in 'video'."},
                        status_code=400,
                    )
                start: int = request_json["video"].get("start", 0)
                end: Optional[int] = request_json["video"].get("end", None)
                mode: SamplingMode = request_json["video"].get(
                    "mode", SamplingMode.FRAMES
                )
                sampling_rate: Optional[float] = request_json["video"].get(
                    "sampling_rate", None
                )
                window_size: Union[int, float] = request_json["video"].get(
                    "window_size", 1
                )
                stride: Union[int, float] = request_json["video"].get("stride", 1)

                dataset = TorchCodecVideoDataset(
                    video_path=video_path,
                    start=start,
                    end=end,
                    mode=mode,
                    window_size=window_size,
                    stride=stride,
                    sampling_rate=sampling_rate,
                    preprocess=self.preprocess,
                )

                batch_size = (
                    self.batch_size
                    if window_size <= 1
                    else int(0.8 * self.batch_size // max(1, int(window_size)))
                )

                data_loader = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.device == "cuda",
                    persistent_workers=self.num_workers > 0,
                    collate_fn=_custom_collate_fn,
                )

                def _run_inference():
                    with torch.no_grad():
                        for i, batch in enumerate(data_loader):
                            frames = batch["frames"].to(self.device)
                            if window_size > 1:
                                batch_embeddings = execute_with_retry(
                                    self.model.encode_video, frames
                                )
                            else:
                                batch_embeddings = execute_with_retry(
                                    self.model.encode_image, frames
                                )
                            meta = batch["meta"]
                            output.append(
                                {
                                    "embeddings": batch_embeddings.cpu().tolist(),
                                    "meta": meta,
                                }
                            )

                await asyncio.to_thread(_run_inference)

            elif "image" in request_json:
                image = request_json["image"]
                image_path = image.get("file_path")
                if image_path is None:
                    raise ValueError(
                        "Request must contain 'file_path' field in 'image'."
                    )
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")

                pil_image = Image.open(image_path).convert("RGB")
                to_tensor_transform = T.ToTensor()
                image_tensor = to_tensor_transform(pil_image)
                image_tensor = (
                    self.preprocess(image_tensor).unsqueeze(0).to(self.device)
                )

                def _run_image_inference():
                    with torch.no_grad():
                        embeddings = execute_with_retry(
                            self.model.encode_image, image_tensor
                        )
                        embeddings = embeddings.cpu().tolist()
                    output.append({"embeddings": embeddings})

                await asyncio.to_thread(_run_image_inference)
            elif "text" in request_json:
                text = request_json["text"]
                tokenizer = transforms.get_text_tokenizer(self.model.context_length)
                text_tensor = tokenizer([text]).to(self.device)

                def _run_text_inference():
                    with torch.no_grad():
                        embeddings = execute_with_retry(
                            self.model.encode_text, text_tensor
                        )
                        embeddings = embeddings.cpu().tolist()
                    output.append({"embeddings": embeddings})

                await asyncio.to_thread(_run_text_inference)

            else:
                raise ValueError(
                    "Request must contain either 'image', 'video' or 'text' field."
                )

            return JSONResponse(
                {
                    "success": True,
                    "output": output,
                },
                status_code=200,
            )

        except Exception as e:

            traceback_str = traceback.format_exc()
            return JSONResponse(
                {
                    "success": False,
                    "error_msg": str(e),
                    "traceback": traceback_str,
                },
                status_code=500,
            )


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments."""

    argparse = ArgumentParser()
    argparse.add_argument("--model_name", type=str, required=True)
    argparse.add_argument("--torch_batch_size", type=int, default=None)

    arg_strings = []
    for key, value in cli_args.items():
        if value is not None:
            arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)

    args = argparse.parse_args(arg_strings)

    return PerceptionEncoderDeployment.bind(  # type: ignore[attr-defined]
        model_name=args.model_name,
        torch_batch_size=args.torch_batch_size,
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import av
import cv2
import numpy as np
import torch
import torchvision.io
import torchvision.transforms.v2 as T
from PIL import Image
from tenacity import RetryError, Retrying, stop_after_attempt, wait_fixed
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder

logger = logging.getLogger("matrix.app_server.vision.utils")
logging.getLogger("httpx").setLevel(logging.WARNING)


class SamplingMode(Enum):
    FRAMES = "frames"
    SECONDS = "seconds"


class TorchCodecVideoDataset(Dataset):
    """
    A unified PyTorch Dataset for loading individual frames or windows (clips)
    of frames from a video using torchcodec.
    """

    def __init__(
        self,
        video_path: str,
        mode: SamplingMode = SamplingMode.FRAMES,
        window_size: Union[int, float] = 1,
        stride: Union[int, float] = 1,
        sampling_rate: float = None,
        start: Union[int, float] = None,
        end: Union[int, float] = None,
        preprocess=None,
        device="cpu",
    ):
        super().__init__()
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        self.video_path = video_path
        self.preprocess = preprocess
        self.device = device
        self.sampling_rate = sampling_rate

        if self.sampling_rate is not None and not 0.0 < self.sampling_rate <= 1.0:
            raise ValueError(f"sampling_rate must be between 0.0 and 1.0")

        decoder = VideoDecoder(self.video_path, device=self.device)
        self.fps = decoder.metadata.average_fps
        self.total_frames = decoder.metadata.num_frames

        if mode == SamplingMode.SECONDS:
            # All inputs are treated as timestamps
            start_frame = 0 if start is None else int(start * self.fps)
            end_frame = self.total_frames - 1 if end is None else int(end * self.fps)
            self.window_size = max(1, int(window_size * self.fps))
            final_stride = max(1, int(stride * self.fps))
        else:
            # All inputs are treated as frame indices
            start_frame = 0 if start is None else int(start)
            end_frame = self.total_frames - 1 if end is None else int(end)
            self.window_size = int(window_size)
            final_stride = int(stride)

        end_frame = int(min(end_frame, self.total_frames - 1))

        if start_frame >= end_frame:
            raise ValueError(
                f"start_frame ({start_frame}) must be less than end_frame ({end_frame})."
            )

        num_frames_in_segment = end_frame - start_frame + 1
        if num_frames_in_segment < self.window_size:
            raise ValueError(
                f"Video segment ({num_frames_in_segment} frames) is smaller than window_size ({self.window_size})."
            )

        # Calculate all possible window/item start positions
        num_items = (num_frames_in_segment - self.window_size) // final_stride + 1
        self.item_start_frames = [
            start_frame + (i * final_stride) for i in range(num_items)
        ]

    def _get_decoder(self):
        """Lazy initializes the decoder for multiprocessing safety."""
        if not hasattr(self, "decoder"):
            self.decoder = VideoDecoder(self.video_path, device=self.device)
        return self.decoder

    def _uniform_sample(self, list_len, num_samples):
        """Helper function to uniformly sample indices from a list."""
        if list_len == 0:
            return []
        if list_len < num_samples:
            indices = np.arange(list_len)
            repeats = num_samples // list_len
            remainder = num_samples % list_len
            indices = np.concatenate([indices] * repeats + [indices[:remainder]])
        else:
            indices = np.linspace(0, list_len - 1, num_samples, dtype=int)
        return indices

    def __len__(self):
        return len(self.item_start_frames)

    def __getitem__(self, idx):
        if not 0 <= idx < len(self.item_start_frames):
            raise IndexError(f"Item index {idx} is out of bounds.")

        decoder = self._get_decoder()

        item_start_frame = self.item_start_frames[idx]
        all_item_indices = list(
            range(item_start_frame, item_start_frame + self.window_size)
        )

        if self.sampling_rate is not None:
            num_to_sample = int(self.window_size * self.sampling_rate)
            sample_idxs_relative = self._uniform_sample(self.window_size, num_to_sample)
            final_indices = [all_item_indices[i] for i in sample_idxs_relative]
            final_indices.sort()
        else:
            final_indices = all_item_indices

        item_tensor = decoder.get_frames_at(final_indices).data

        if self.preprocess:
            processed_frames = [self.preprocess(frame) for frame in item_tensor]
            item_tensor = torch.stack(processed_frames)
        metadata = {
            "frame_indices": final_indices,
            "timestamps_sec": [i / self.fps for i in final_indices],
            "source_fps": self.fps,
            "window_idx": idx,
            "total_frames": self.total_frames,
        }

        # If window_size is 1, return a single frame tensor, not a batch of 1
        if self.window_size == 1 and item_tensor.shape[0] == 1:
            return {"frames": item_tensor.squeeze(0), "meta": metadata}

        return {"frames": item_tensor, "meta": metadata}


def get_image_transform(
    image_size: int,
    center_crop: bool = False,
    interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
):
    if center_crop:
        crop = [
            T.Resize(image_size, interpolation=interpolation),
            T.CenterCrop(image_size),
        ]
    else:
        # "Squash": most versatile
        crop = [T.Resize((image_size, image_size), interpolation=interpolation)]

    # This pipeline expects a (C, H, W) tensor of type uint8 from the decoder
    return T.Compose(
        crop
        + [
            # 1. Convert the uint8 tensor to a float tensor and scale values to [0.0, 1.0]
            T.ToDtype(torch.float32, scale=True),
            # 2. Normalize the float tensor to the range [-1.0, 1.0]
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
        ]
    )


def execute_with_retry(func, *args, **kwargs):

    retryer = Retrying(
        stop=stop_after_attempt(4),  # Max 4 attempts
        wait=wait_fixed(0.5),  # Wait for 0.5 seconds before retrying
        reraise=True,  # If it fails, re-raise the last exception
    )
    return retryer(func, *args, **kwargs)

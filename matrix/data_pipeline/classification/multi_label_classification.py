# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import json
import os
import re
import socket
import time
import typing as tp

import numpy as np
import ray
import torch
from fire import Fire
from transformers import pipeline


@dataclasses.dataclass
class TextClassificationConfig:
    model_dir: str
    text_key: str
    threshold_fname: str = "thresholds.jsonl"
    unclassified_label: str = "unclassified"
    top_k: str = "3"


class TextClassification:
    def __init__(self, config: TextClassificationConfig, device: str = "cuda:0"):
        self.cfg = config
        self.device = device
        self.setup()

    def setup(self) -> None:
        print(f"actor hostname is {socket.gethostname()}")
        assert torch.cuda.is_available(), "CUDA is not available."

        model_local_dir = self.cfg.model_dir
        self.classifier = pipeline(
            "text-classification",
            model=model_local_dir,
            tokenizer=model_local_dir,
            device=self.device,
            top_k=int(self.cfg.top_k),
        )
        if self.cfg.threshold_fname and os.path.exists(self.cfg.threshold_fname):
            with open(
                os.path.join(model_local_dir, self.cfg.threshold_fname), "r"
            ) as f:
                self.thresholds = json.loads(f.readline())
            self.labels: tp.List[str] = sorted(self.thresholds.keys())
        else:
            self.labels = list(self.classifier.model.config.id2label.values())
            self.thresholds = {label: 0.5 for label in self.labels}

    def __call__(self, samples: tp.Dict[str, np.ndarray]) -> tp.Dict[str, np.ndarray]:
        cleaned_text = []
        for data in samples[self.cfg.text_key]:
            match = re.search(
                r"\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", data, re.DOTALL
            )
            if match:
                data = match.group(1)
            cleaned_text.append(data)
        predictions = self.classifier(
            cleaned_text,
            truncation=True,
        )
        valid_labels: tp.List[tp.List[str]] = []
        valid_label_scores: tp.List[tp.List[float]] = []
        for prediction in predictions:
            _valid_labels: tp.List[tp.Tuple[str, float]] = []
            for label_score in prediction:
                label = label_score["label"]  # type: ignore[index]
                score = float(label_score["score"])  # type: ignore[index]
                if score >= self.thresholds.get(label, 0):
                    _valid_labels.append((label, score))
            if _valid_labels:
                _valid_labels.sort(key=lambda x: x[1], reverse=True)
                valid_labels.append([x[0] for x in _valid_labels])
                valid_label_scores.append([float(x[1]) for x in _valid_labels])
            else:
                valid_labels.append([self.cfg.unclassified_label])
                valid_label_scores.append([float(-1.0)])
        samples["valid_labels"] = valid_labels  # type: ignore[assignment]
        samples["valid_label_scores"] = valid_label_scores  # type: ignore[assignment]
        return samples


# don't run locally https://github.com/ray-project/ray/issues/35537
@ray.remote
def run_remotely(
    config: TextClassificationConfig,
    input_jsonl: str,
    output_dir: str,
    batch_size: int,
    max_concurrency: int,
):
    print(f"driver hostname is {socket.gethostname()}")
    ds = ray.data.read_json(input_jsonl)  # type: ignore[attr-defined]
    res_datasets = ds.map_batches(
        TextClassification,
        batch_size=batch_size,
        num_gpus=1,
        concurrency=max_concurrency,
        fn_constructor_kwargs={"config": config},
    )

    res_datasets.write_json(output_dir, force_ascii=False)


def main(
    ray_head_url: str,
    model: str,
    input_jsonl: str,
    output_dir: str,
    batch_size: int = 1024,
    max_concurrency: int = 8,
    text_key: str = "src",
    **kwargs,
):
    """Run knowledge classifier on input jsonl.
    params:
    ray_head_url: Ray head (hostname:client_server_port), eg localhost:10001
    output_dir: name of the output directory.
    input_jsonl: file or dir of input jsonl.
    model: the huggingface model name or a directory.
    batch_size: request batching.
    max_concurrency: num of gpus for inference.
    """
    assert os.path.exists(input_jsonl), f"{input_jsonl} does not exist."
    assert not os.path.exists(output_dir), f"{output_dir} already exists."
    assert ":" in ray_head_url, "ray_head_url should be in the format of hostname:port"
    if not ray_head_url.startswith("ray://"):
        ray_head_url = f"ray://{ray_head_url}"

    config_params = {
        "model_dir": model,
        "text_key": text_key,
    }

    # Only add kwargs that are fields in TextClassificationConfig
    valid_fields = {f.name for f in dataclasses.fields(TextClassificationConfig)}
    for key, value in kwargs.items():
        if key in valid_fields:
            config_params[key] = value

    config = TextClassificationConfig(**config_params)

    ray.init(address=ray_head_url, log_to_driver=True)
    start_time = time.time()
    ray.get(
        run_remotely.remote(
            config, input_jsonl, output_dir, batch_size, max_concurrency
        )
    )
    print(f"Time taken: {time.time() - start_time} seconds")


if __name__ == "__main__":
    Fire(main)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import re
import shlex
import threading
import time
from collections import defaultdict

import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Checkpoint evaluation constants and helper functions
DEFAULT_CONFIG = [
    "--shot",
    "0",
    "--tag",
    "--cot",
    "direct",
]

BENCHMARK_CONFIG = {
    "gpqa_diamond": ["--dataset", "gpqa", "--diamond"],
    "math_500": ["--dataset", "math", "--math_500"],
    "mmlu_pro": ["--dataset", "mmlu_pro"],
    "super_gpqa": ["--dataset", "gpqa", "--super"],
}

DEFAULT_NUM_SEEDS = {
    "gpqa_diamond": 24,
    "math_500": 16,
    "mmlu_pro": 1,
    "super_gpqa": 1,
}


def extract_benchmark_data(data_dict, metric_pattern):
    """
    Extract benchmark information from the provided dictionary.
    """
    results: defaultdict[str, dict] = defaultdict(
        lambda: {
            "successes": 0,
            "pending": 0,
            "failures": 0,
            "metric_values": [],
            "metric_avg": 0.0,
            "metric_stderr": 0.0,
        }
    )

    # Compile regex patterns
    metric_regex = re.compile(metric_pattern)
    for key, value in data_dict.items():
        benchmark_match = re.match(r"(.+?)_seed\d+", key)
        if not benchmark_match:
            logger.warning(f"Format should be benchmark_seed_with_num: {key}")
            continue

        benchmark = benchmark_match.group(1)

        if "success" not in value:
            logger.warning(f"Wrong result format for {key}: {value}")
            continue

        success = value.get("success", False)
        stdout = value.get("stdout", "")
        if success:
            metric_match = metric_regex.search(stdout)
            if not metric_match:
                logger.warning(f"No metric found in stdout for {key} from {stdout}")
                continue

            metric_value = float(metric_match.group(1))
            results[benchmark]["successes"] += 1
            results[benchmark]["metric_values"].append(metric_value)
        elif value.get("status", "UNKNOWN") not in ["FAILED", "SUCCEEDED"]:
            results[benchmark]["pending"] += 1
        else:
            results[benchmark]["failures"] += 1

    # Calculate average metrics
    for benchmark, data in results.items():
        if data["metric_values"]:
            data["metric_avg"] = sum(data["metric_values"]) / len(data["metric_values"])
            data["metric_stderr"] = np.std(data["metric_values"]) / np.sqrt(
                len(data["metric_values"])
            )

    return dict(results)


def run_eval_script(
    pythonpath: str,
    eval_script: str,
    eval_save_dir: str,
    checkpoint_dir: str,
    benchmark: str,
    seed: int,
    thinking: bool,
    cluster_id: str,
    matrix_dir: str,
    app_name: str,
    use_ray_data: bool,
    ray_head_address: str,
    tokenizer: str,
    sampling_params: dict | None = None,
    skip_generation=False,
):
    """Generate environment and command for evaluation script."""
    env = {"PYTHONPATH": pythonpath} if pythonpath else {}
    if eval_save_dir.startswith("s3://"):
        cache_dir = os.environ.get(
            "MATRIX_CACHE", os.path.expanduser("~/.cache/matrix")
        )
        upload_eval_dir = eval_save_dir
        eval_save_dir = os.path.join(
            cache_dir, "evaluation", eval_save_dir[len("s3://") :]
        )
    else:
        upload_eval_dir = ""
    command = (
        [
            "python",
            eval_script,
            "--eval_save_dir",
            eval_save_dir,
            "--ext_model",
            checkpoint_dir,
            "--seed",
            str(seed),
        ]
        + (
            [
                "--matrix",
                '{"cluster_id": cluster_id, "matrix_dir": matrix_dir, "app_name": app_name}',
            ]
            if not use_ray_data
            else ["--ray_data", ray_head_address]
        )
        + (
            [
                "--upload_eval_dir",
                upload_eval_dir,
            ]
            if upload_eval_dir
            else []
        )
        + DEFAULT_CONFIG
        + BENCHMARK_CONFIG[benchmark]
        + (["--thinking"] if thinking else [])
        + [
            "--tokenizer",
            tokenizer,
        ]
        + (
            [
                "--sampling_params",
                shlex.quote(json.dumps(sampling_params)),
            ]
            if sampling_params
            else []
        )
        + (
            [
                "--skip_generation",
            ]
            if skip_generation
            else []
        )
    )

    return env, " ".join(command)

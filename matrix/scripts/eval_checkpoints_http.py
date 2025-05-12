# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
import runpy
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

from fire import Fire

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import time
import uuid
from urllib.parse import urljoin

import requests


def main(
    matrix_http_server: str,
    checkpoint_dir: str,
    eval_save_dir: str,
    model_replica: int = 8,
    thinking: bool = True,
    job_id: str | None = None,
    benchmarks: list[str] | None = None,
    num_seeds: int | None = None,
    max_concurrent_tasks: int = 3,
    timeout: int = 36000,
    model_size: str = "8B",
    tokenizer: str = "meta-llama/Llama-3.1-8B-Instruct",
):
    job_id = (
        job_id or f"{checkpoint_dir.strip('/').split('/')[-1]}-{uuid.uuid4().hex[:8]}"
    )
    post_url = urljoin(matrix_http_server, "/checkpoint-eval")

    payload = {
        "checkpoint_dir": checkpoint_dir,
        "eval_save_dir": eval_save_dir,
        "model_replica": model_replica,
        "max_concurrent_tasks": max_concurrent_tasks,
        "model_size": model_size,
        "tokenizer": tokenizer,
        "thinking": thinking,
        "timeout": timeout,
    }
    if job_id:
        payload["job_id"] = job_id
    if benchmarks:
        payload["benchmarks"] = benchmarks
    if num_seeds:
        payload["num_seeds"] = num_seeds

    resp = requests.post(post_url, json=payload)
    data = resp.json()
    assert resp.ok, f"Request failed: {data.get('detail', data)}"
    job_id = data["job_id"]
    print(f"[INFO] Submitting eval job to {post_url} with job_id={job_id}")

    status_url = urljoin(matrix_http_server, f"/jobs/{job_id}/status")
    metrics_url = urljoin(matrix_http_server, f"/checkpoint-eval/{job_id}/metrics")
    print(f"[INFO] Job submitted. Polling status at {status_url}...")
    while True:
        resp = requests.get(status_url)
        data = resp.json()  # parse first
        assert resp.ok, f"Request failed: {data.get('detail', data)}"
        print(f"[STATUS] Job status: {data}")
        if data["status"] in ["COMPLETED", "FAILED"]:
            done = True
            break
        time.sleep(30)

    print(f"[INFO] Job finished. Fetching metrics from {metrics_url}")
    resp = requests.get(metrics_url)
    data = resp.json()  # parse first
    assert resp.ok, f"Request failed: {data.get('detail', data)}"
    print(data)


if __name__ == "__main__":
    Fire(main)

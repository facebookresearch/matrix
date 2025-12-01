# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shlex
import subprocess
import time

import fire
import tqdm
from datasets import load_dataset


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def download(
    dataset_name: str = "princeton-nlp/SWE-Bench_Verified",
    split: str = "test",
    output_dir: str = os.path.expanduser("~/checkpoint/cache/apptainer_sif"),
    sleep_seconds: int = 10 * 60,
):
    os.makedirs(output_dir, exist_ok=True)
    dataset_val = load_dataset(dataset_name, split=split).to_list()
    for item in tqdm.tqdm(dataset_val):
        image = get_swebench_docker_image_name(item)
        filename = os.path.basename(image).replace(":", "_") + ".sif"
        filename = os.path.join(output_dir, filename)
        if not os.path.exists(filename):
            try:
                subprocess.run(
                    ["apptainer", "pull", filename, f"docker://{image}"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except Exception as e:
                estderr = e.stderr if hasattr(e, "stderr") else None
                print(f"{image}: {repr(e)} {estderr}")  # type: ignore[attr-defined]
                time.sleep(sleep_seconds)


if __name__ == "__main__":
    fire.Fire(download)

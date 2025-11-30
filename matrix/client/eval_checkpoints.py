# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import time
import typing as tp
from pathlib import Path

from fire import Fire

import matrix
import matrix.utils.os
from matrix import Cli
from matrix.job.job_api import JobApi


def main(
    checkpoints: tp.List[tp.Dict[str, tp.Any]],
    model_args: tp.Dict[str, tp.Any],
    command: str,
    job_apps: tp.List[tp.Dict[str, tp.Any]] = [],
    **job_kwargs,
):
    """
    command: the command template, to be formated by checkpoint.

     Args:
        checkpoints: A list of checkpoint dictionaries. Each dictionary
            must contain 'name' and 'path' keys. The 'name' is used for the
            application name within the task, and 'path' is used as the
            model name for the application.
        model_args: Additional arguments to be included in the application
            definition for each checkpoint's task.
        command: The command template string to execute for each checkpoint.
            It will be formatted using the corresponding checkpoint dictionary
            (e.g., allowing use of '{name}', '{path}').
        job_apps: A list of additional application dictionaries to include
            in the main job definition. Each dictionary must contain a 'name'
            key. These are static applications independent of the checkpoints,
            such as extractors and judge LLM. Defaults to [].
        **job_kwargs: Additional keyword arguments to be included in the
            main job definition (e.g., 'max_concurrent_tasks', 'timeout').

    Example:
    python -m matrix.client.eval_checkpoints \
    --checkpoints "[{'name': 'step_1000', 'path': '/checkpoints/step_1000/'}, {'name': 'step_200', 'path': '/checkpoints/step_200'}]" \
    --model_args "{'tokenizer': 'meta-llama/Llama-3.1-8B-Instruct', 'model_size': '8B'}" \
    --job_apps "[{'model_name': 'meta-llama/Llama-3.1-8B-Instruct', 'use_grpc': 'true', 'model_size': '8B', 'name': '8B_grpc'}]" \
    --command "matrix llm_inference --app_name {name} --input_jsonls /math-500/test.jsonl --output_jsonl /tmp/{name}.jsonl --batch_size=64 --system_prompt 'Please reason step by step, and put your final answer within \boxed{{}}.' --max_tokens 30000 --text_key problem --timeout_secs 1800 --override_output_file True" \
    --max_concurrent_tasks 2
    """
    for app in job_apps:
        assert "name" in app, "name are required for each app"
    for cp in checkpoints:
        assert "name" in cp, f"name is missing in checkpoint {cp}"
        assert "path" in cp, f"path is missing in checkpoint {cp}"

    cli = Cli()
    task_definitions = [
        {
            "applications": [
                {
                    "name": cp_info["name"],
                    "model_name": cp_info["path"],
                    **model_args,
                }
            ],
            "func": functools.partial(matrix.utils.os.run_and_stream, blocking=True),
            "kwargs": {
                "command": command.format(**cp_info),
            },
        }
        for cp_info in checkpoints
    ]
    job_def = {
        "applications": job_apps,
        "task_definitions": task_definitions,
        **job_kwargs,
    }
    job_id = cli.job.submit(job_def)
    while True:
        status = cli.job.status(job_id)
        print(status)
        if status["status"] in ["COMPLETED", "FAILED"]:
            break
        time.sleep(30)
    results = cli.job.get_results(job_id)
    print(results)


if __name__ == "__main__":
    Fire(main)

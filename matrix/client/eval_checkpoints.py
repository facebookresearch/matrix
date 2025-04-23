import functools
import time
import typing as tp
from pathlib import Path

from fire import Fire

import matrix
from matrix import Cli
from matrix.job.job_api import JobApi


def main(
    checkpoints: tp.List[tp.Dict[str, tp.Any]],
    model_size: str,
    tokenizer: Path,
    job_apps: tp.List[tp.Dict[str, tp.Any]],
    command: str,
    **job_kwargs,
):
    """
    command: the command template, to be formated by checkpoint.
    """
    print(command)
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
                    "model_size": model_size,
                    "tokenizer": str(tokenizer),
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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# lighteval > 0.10.0 is required
# pip install "git+https://github.com/huggingface/lighteval.git#egg=lighteval[litellm]"

import time
import typing as tp

import fire
import lighteval
import yaml
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.endpoints.litellm_model import LiteLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

import matrix
import matrix.utils.ray


def main(
    cluster_id="lighteval_test",
    model_name="/datasets/pretrained-llms/Llama-3.1-8B-Instruct",
    num_replicas=8,
    app_name="8B",
    slurm_account="data",
    slurm_qos="data_high",
    eval_task="lighteval|math_500|0",
    max_samples: int | None = None,
    cleanup=False,
    custom_tasks_directory: str | None = None,
    generation_parameters: dict[str, tp.Any] = None,
):

    # default generation parameters
    default_generation_parameters: dict[str, tp.Any] = {
        "temperature": 0.6,
        "max_new_tokens": 16384,
        "top_p": 0.95,
        "seed": 42,
        "repetition_penalty": 1.0,
        "frequency_penalty": 0.0,
    }

    # override defaults if user passes something in
    if generation_parameters is not None:
        default_generation_parameters.update(generation_parameters)

    def setup():
        cli = matrix.Cli(
            cluster_id=cluster_id,
        )
        num_gpu = 0
        try:
            resources = cli.cluster.get_resources()
            num_gpu = resources["available_resources"].get("GPU", 0)
        except:
            pass

        if num_gpu == 0:
            cli.start_cluster(
                add_workers=1,
                slurm={
                    "account": slurm_account,
                    "qos": slurm_qos,
                },
            )

        try:
            cli.get_app_metadata(app_name)
        except Exception as e:
            if "uknown app_name" in str(e):
                cli.deploy_applications(
                    applications=[
                        {
                            "model_name": model_name,
                            "min_replica": num_replicas,
                            "name": app_name,
                            "model_size": "8B",
                        }
                    ]
                )
        while (status := cli.app.app_status(app_name)) != "RUNNING":
            print(f"{app_name} not ready, current status {status}")
            time.sleep(10)
        base_url = cli.get_app_metadata(app_name)["endpoints"]["head"]
        return cli, base_url

    # https://huggingface.co/docs/lighteval/en/using-the-python-api
    def run_eval(base_url):
        evaluation_tracker = EvaluationTracker(
            output_dir="./results",
            save_details=True,
            push_to_hub=False,
        )

        pipeline_params = PipelineParameters(
            launcher_type=ParallelismManager.OPENAI,
            max_samples=max_samples,
            custom_tasks_directory=custom_tasks_directory,
        )

        yaml_str = f"""
        model_parameters:
            model_name: "openai/{model_name}"
            provider: "openai"
            base_url: {base_url}
            api_key: "EMPTY"
            generation_parameters: {default_generation_parameters}
        """
        data: dict = yaml.safe_load(yaml_str)
        print("lighteval config data:", data)

        model_config = LiteLLMModelConfig(**data["model_parameters"])

        pipeline = Pipeline(
            tasks=eval_task,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model_config=model_config,
        )

        pipeline.evaluate()
        pipeline.save_and_push_results()
        pipeline.show_results()

    cli = None
    try:
        cli, base_url = setup()
        run_eval(base_url)
    finally:
        if cleanup and cli is not None:
            cli.deploy_applications(
                action=matrix.utils.ray.Action.REMOVE,
                applications=[
                    {
                        "name": app_name,
                    }
                ],
            )

            cli.stop_cluster()


if __name__ == "__main__":
    fire.Fire(main)

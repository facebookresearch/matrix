# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import socket
import time
import typing as tp

import numpy as np
import ray
from fire import Fire
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

USER_MESSAGE = "<user_message>"


class VllmInference:
    def __init__(
        self,
        text_key: str,
        model_args: dict,
        sampling_params: dict,
        prompt_template: str,
        output_key: str,
    ) -> None:

        self.vllm_model = LLM(**model_args)
        self.sampling_parameters = SamplingParams(**sampling_params)
        self.text_key = text_key
        self.prompt_template = prompt_template
        self.output_key = output_key

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Processes a batch of prompts using the initialized vLLM model.
        This method is called by Ray Data for each batch.
        """
        # Extract prompts from the input batch dictionary
        prompts = batch[self.text_key].tolist()
        if self.prompt_template:
            prompts = [
                self.prompt_template.replace(USER_MESSAGE, prompt) for prompt in prompts
            ]
        outputs = self.vllm_model.generate(
            prompts, sampling_params=self.sampling_parameters, use_tqdm=False
        )

        responses = [[element.text for element in output.outputs] for output in outputs]
        result = dict(batch)  # shallow copy of original batch dict
        result[self.output_key] = np.array(responses)
        return result


# don't run locally https://github.com/ray-project/ray/issues/35537
@ray.remote
def run_remotely(
    input_jsonl: str,
    output_dir: str,
    text_key: str,
    batch_size: int,
    model_args: dict,
    sampling_params: dict,
    prompt_template: str,
    min_concurrency: int,
    output_key: str,
):
    logger.info(f"driver hostname is {socket.gethostname()}")

    model_args.update({"distributed_executor_backend": "mp"})
    total_gpus = ray.cluster_resources().get("GPU", 0)
    tensor_parallel_size = model_args.get("tensor_parallel_size", 1)
    input_dataset = ray.data.read_json(input_jsonl)
    count = input_dataset.count()
    input_dataset = input_dataset.zip(
        ray.data.range(count).map(lambda i: {"line_number": i["id"]})
    )

    max_concurrency = max(
        1, min(int(total_gpus // tensor_parallel_size), int(count // batch_size))
    )
    min_concurrency = min(min_concurrency, max_concurrency)
    logger.info(
        f"min_concurrency: {min_concurrency}, max_concurrency: {max_concurrency}"
    )

    response = input_dataset.map_batches(
        VllmInference,
        fn_constructor_kwargs={
            "text_key": text_key,
            "model_args": model_args,
            "sampling_params": sampling_params,
            "prompt_template": prompt_template,
            "output_key": output_key,
        },
        batch_size=batch_size,
        num_gpus=tensor_parallel_size,
        concurrency=(min_concurrency, max_concurrency),
    )
    response.write_json(output_dir, force_ascii=False)


def main(
    ray_head_url: str,
    input_jsonl: str,
    output_dir: str,
    text_key: str = "problem",
    batch_size: int = 32,
    model_args: dict[str, tp.Any] = {},
    sampling_params: dict[str, tp.Any] = {},
    prompt_template: str = "",
    min_concurrency: int = 1,
    output_key: str = "response",
):
    """Run llm generate on input jsonl prompt.
    params:
    ray_head_url: Ray head (hostname:client_server_port), eg localhost:10001
    input_jsonl: file or dir of input jsonl.
    output_dir: name of the output directory.
    text_key: the text field in the input json for user question.
    batch_size: size of batch inference.
    model_args: vllm model args https://docs.vllm.ai/en/v0.8.3/api/offline_inference/llm.html.
    sampling_params: sampling parameters https://docs.vllm.ai/en/latest/api/vllm/vllm.sampling_params.html#vllm.sampling_params.SamplingParams
    prompt_template: template to convert user question (coded as <user_message>) into llm prompt.
    min_concurrency: minimum num of concurrent tasks.
    """
    assert "model" in model_args, f"model is missing from {model_args}"
    assert os.path.exists(input_jsonl), f"{input_jsonl} does not exist."
    assert ":" in ray_head_url, "ray_head_url should be in the format of hostname:port"
    if not ray_head_url.startswith("ray://"):
        ray_head_url = f"ray://{ray_head_url}"

    logger.info(f"Prompt: {prompt_template}")
    ray.init(
        address=ray_head_url,
        runtime_env={
            "env_vars": {
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "VLLM_CONFIGURE_LOGGING": "0",
            }
        },
    )
    start_time = time.time()
    ray.get(
        run_remotely.remote(
            input_jsonl,
            output_dir,
            text_key,
            batch_size,
            model_args,
            sampling_params,
            prompt_template,
            min_concurrency,
            output_key,
        )
    )
    print(f"Time taken: {time.time() - start_time} seconds")


if __name__ == "__main__":
    Fire(main)

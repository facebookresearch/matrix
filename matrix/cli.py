# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import asyncio
import getpass
import json
import os
import subprocess
import time
import typing as tp
from pathlib import Path

import fire

from matrix.app_server import app_api
from matrix.cluster.ray_cluster import RayCluster
from matrix.utils.basics import convert_to_json_compatible
from matrix.utils.os import run_and_stream, run_async, run_subprocess


class Cli:
    """
    Matrix command-line interface tool.

    Matrix CLI provides a unified interface to manage Ray clusters and deploy applications
    for synthetic data generation. It supports large-scale inference using vLLM, proxy servers
    for various LLM providers, and code execution services.

    The CLI allows users to:
    - Start and stop Ray clusters
    - Check cluster status
    - Deploy applications (vLLM serving, LLM proxy, code execution)
    - Run LLM inference
    - Query application metadata
    - Test deployed applications
    """

    cluster_id: tp.Optional[str]
    cluster: RayCluster

    def __init__(
        self, cluster_id: tp.Optional[str] = None, matrix_dir: tp.Optional[str] = None
    ):
        """
        Initializes the Matrix CLI by connecting to an existing cluster or preparing for a new one.

        The CLI uses either provided parameters, environment variables, or default values to
        determine the cluster ID and matrix directory for saving files.

        Args:
            cluster_id (str, optional): The unique identifier for the Ray cluster.
                If not provided, uses MATRIX_CLUSTER_ID environment variable or
                defaults to username + "_cluster".
            matrix_dir (str, optional): The directory for saving files.
                If not provided, uses MATRIX_DIR environment variable or
                defaults to ~/.matrix.
        """
        self.cluster_id = (
            cluster_id
            or os.getenv("MATRIX_CLUSTER_ID")
            or (getpass.getuser() + "_cluster")
        )
        self.matrix_dir = Path(
            matrix_dir
            if matrix_dir
            else (os.getenv("MATRIX_DIR") or str(Path.home() / ".matrix"))
        )
        self.cluster = RayCluster(
            cluster_id=self.cluster_id,
            matrix_dir=self.matrix_dir,
        )

    def start_cluster(
        self,
        add_workers: int = 0,
        slurm: tp.Dict[str, tp.Union[str, int]] | None = None,
        local: tp.Dict[str, tp.Union[str, int]] | None = None,
        enable_grafana: bool = False,
        force_new_head: bool = False,
    ) -> tp.Dict[str, tp.Any]:
        """
        Starts the Ray cluster with additional keyword arguments. Only do this for new clusters.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the RayCluster's start_head method.
        """
        """
        Starts the Ray cluster with the specified number of workers and additional configuration.
        
        Can add additional workers if the cluster already exists.
        
        Args:
            add_workers (int): Number of worker nodes to add in the cluster.
            slurm (dict, optional): resources for slurm cluster.
            local (dict, optional): resources for local cluster.
            enable_grafana (bool, optional): If True, enable prometheus and grafana dashboard.
            force_new_head (bool): force to remove head.json if haven't run 'matrix stop_cluster'.
            
        Returns:
            None
        """
        status = self.cluster.start(
            add_workers,
            slurm,
            local,
            enable_grafana=enable_grafana,
            force_new_head=force_new_head,
        )
        return convert_to_json_compatible(status)

    def stop_cluster(self):
        """
        Shuts down the Ray cluster.

        This command gracefully terminates all processes in the Ray cluster,
        releasing resources back to the system or Slurm allocation.

        Returns:
            None
        """
        self.cluster.stop()

    def status(self, replica=False) -> tp.List[str]:
        """
        Prints the status of the Ray cluster and deployed applications.

        Displays information about the head node, SSH connection details, and
        runs 'ray status' command to show cluster information. Also shows the
        status of deployed Serve applications.

        Args:
            replica (bool, optional): If True, shows detailed status including replicas.
                Defaults to False.

        Returns:
            None
        """
        head = self.cluster.cluster_info()
        if not head:
            return ["head not started"]
        else:
            assert head.hostname
            results = []
            results.append(
                f"ssh to head node:\nssh -L {head.dashboard_port}:localhost:{head.dashboard_port} -L {head.prometheus_port}:localhost:{head.prometheus_port} -L {head.grafana_port}:localhost:{head.grafana_port} {head.hostname}"
            )  # type: ignore[union-attr]
            cluster_info = convert_to_json_compatible(head)
            results.append(f"Head Info: {json.dumps(cluster_info, indent=2)}")

            results.append("\nRay status: --------")
            ray_status = run_and_stream(
                {},
                " ".join(
                    ["ray", "status", "--address", f"{head.hostname}:{head.port}"]
                ),
                blocking=True,
                return_stdout_lines=1000,
            )
            results.extend(
                ray_status.get("stdout", ray_status.get("error", "")).split("\n")
            )
            results.append("\n\nServe status: --------")
            results.extend(self.app.status(replica))
        return results

    def deploy_applications(
        self,
        action: str | app_api.Action = app_api.Action.REPLACE,
        applications: tp.Optional[tp.List[tp.Dict[str, tp.Union[str, int]]]] = None,
        yaml_config: tp.Optional[str] = None,
    ):
        """
        Deploy applications on top of the Ray cluster.

        This method can deploy various applications such as vLLM serving,
        LLM proxies, or code execution services to the Ray cluster.

        Args:
            action (str | Action, optional): The deployment action to perform.
                Can be REPLACE, REMOVE, or ADD. Defaults to REPLACE.
            applications (List[Dict], optional): List of application configurations.
                Each dictionary should contain application specifications.
            yaml_config (str, optional): Path to a YAML file containing application
                configurations. Used as an alternative to the applications parameter.

        Returns:
            The deployed application names.
        """
        return self.app.deploy(
            action,
            applications,
            yaml_config,
        )

    def inference(
        self,
        app_name: str,
        output_jsonl: str,
        input_jsonls: str | None = None,
        input_hf_dataset: str | None = None,
        hf_dataset_split: str = "train",
        **kwargs,
    ):
        """
        Run batch inference using a deployed application.

        This method processes input data through a deployed application and
        saves the results to the specified output file.

        Args:
            app_name (str): The name of the deployed application to use.
            output_jsonl (str): Path to save inference results in JSONL format.
            input_jsonls (str | None): Path to input data in JSONL format.
            input_hf_dataset (str | None): Hugging Face dataset name to load directly.
            hf_dataset_split (str): Dataset split to load when using a Hugging Face dataset.
            **kwargs: Additional parameters for inference (e.g., temperature, max_tokens).

        Returns:
            None
        """
        return self.app.inference(
            app_name,
            output_jsonl,
            input_jsonls,
            input_hf_dataset=input_hf_dataset,
            hf_dataset_split=hf_dataset_split,
            **kwargs,
        )

    def get_app_metadata(
        self,
        app_name: str,
        endpoint_ttl_sec: int = 5,
        model_name: str | None = None,
        head_only: bool = False,
    ) -> tp.Dict[str, tp.Any]:
        """
        Retrieve metadata for a deployed application.

        This method returns configuration and status information about a deployed
        application, which can be useful for debugging or monitoring.

        Args:
            app_name (str): The name of the deployed application.
            endpoint_ttl_sec (int, optional): Endpoint time-to-live in seconds.
                Defaults to 5.
            model_name (str, optional): Specific model name to query if the app_name is missing.
            head_only (bool, optional): If True, only returns metadata from the head node.
                Defaults to False.

        Returns:
            The application metadata.
        """
        return self.app.get_app_metadata(
            app_name,
            endpoint_ttl_sec=endpoint_ttl_sec,
            model_name=model_name,
            head_only=head_only,
        )

    def check_health(
        self,
        app_name: str,
        prompt: str | None = None,
        use_curl: bool = True,
        use_chat: bool = True,
        use_tools: bool = False,
        use_image: bool = False,
        **kwargs,
    ) -> bool:
        """
        Test a deployed application with an optional prompt.

        This method allows testing of deployed applications by sending a prompt
        and displaying the response. It supports both curl-based requests and
        native chat interfaces.

        Args:
            app_name (str): The name of the deployed application to test.
            prompt (str, optional): The prompt to send to the application.
                If None, a default prompt is used.
            use_curl (bool, optional): If True, uses curl for the test request.
                Defaults to True.
            use_chat (bool, optional): If True, uses chat format for the request.
                Defaults to True.
            use_tools (bool, optional): If True, enables tool usage in the request.
            **kwargs: Additional parameters for the test request.

            Returns:
                True if app is healthy.
        """
        head = self.cluster.cluster_info()
        if not head:
            print("head not started")
            return False
        else:
            assert head.hostname
            metadata = self.get_app_metadata(app_name)
            if not metadata:
                return False
            deployment_name = metadata["deployment_name"]

            if "code" in deployment_name.lower():
                url = f"{metadata['endpoints']['head']}"
                code = "import numpy as np\nprint(np.ones((2,3)).sum())"
                data_json = json.dumps(
                    {
                        "code": code,
                        "timeout": 10,
                    }
                )
                curl_command = [
                    "curl",
                    url,
                    "-H",
                    "Content-Type: application/json",
                    "-d",
                    data_json,
                ]
                return run_subprocess(curl_command)
            elif "hello" in deployment_name.lower():
                url = f"{metadata['endpoints']['head']}"
                curl_command = [
                    "curl",
                    url,
                ]
                return run_subprocess(curl_command)
            elif "container" in deployment_name.lower():

                async def run_container():
                    from matrix.client.container_client import (
                        ContainerClient,
                        ManagedContainer,
                    )

                    if metadata["args"].get("ray_resources", {}).get("num_gpus", 0) > 0:
                        async with ManagedContainer(
                            metadata["endpoints"]["head"],
                            image="docker://pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime",
                            run_args=["--nv"],
                        ) as client:
                            return await client.execute(
                                [
                                    "python",
                                    "-c",
                                    "import torch; "
                                    "print('CUDA available:', torch.cuda.is_available()); "
                                    "print('Device count:', torch.cuda.device_count()); "
                                    "print(torch.randn(2,3).cuda())",
                                ]
                            )
                    else:
                        async with ManagedContainer(
                            metadata["endpoints"]["head"],
                            image="docker://ubuntu:22.04",
                        ) as client:
                            return await client.execute(["echo", "Hello World"])

                return run_async(run_container())
            else:
                if not prompt:
                    if use_image:
                        prompt = "Read all the text in the image."
                    elif use_tools:
                        prompt = "Get the weather in SF using the given tool"
                    else:
                        prompt = "What is 2+4=?"
                tools = (
                    [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get the current weather in a given location",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "location": {
                                            "type": "string",
                                            "description": "City and state, e.g., 'San Francisco, CA'",
                                        },
                                        "unit": {
                                            "type": "string",
                                            "enum": ["celsius", "fahrenheit"],
                                        },
                                    },
                                    "required": ["location", "unit"],
                                },
                            },
                        }
                    ]
                    if use_tools
                    else None
                )
                tool_choice = "auto" if use_tools else None
                data_payload = {
                    "model": metadata["model_name"],
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    "temperature": 0.7,
                }
                if use_image:
                    text_prompt = data_payload["messages"][1]["content"]
                    data_payload["messages"][1]["content"] = [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
                            },
                        },
                        {"type": "text", "text": text_prompt},
                    ]
                if use_tools:
                    data_payload |= {
                        "tools": tools,
                        "tool_choice": tool_choice,
                    }
                if use_curl and not metadata["use_grpc"]:
                    url = f"{metadata['endpoints']['head']}/chat/completions"
                    data_json = json.dumps(data_payload)
                    curl_command = [
                        "curl",
                        url,
                        "-H",
                        "Content-Type: application/json",
                        "-d",
                        data_json,
                    ]
                    return run_subprocess(curl_command)
                else:
                    from matrix.client import query_llm

                    data_payload.pop("tools", None)
                    data_payload.pop("tool_choice", None)
                    if not use_chat:
                        data_payload = {"prompt": prompt}
                    response = query_llm.batch_requests(
                        metadata["endpoints"]["head"],
                        metadata["model_name"],
                        [data_payload],
                        app_name=app_name,
                        tool_choice=tool_choice,
                        tools=tools,
                        **kwargs,
                    )[0]
                    print(response)

                    only_response: dict[str, tp.Any] = response["response"]  # type: ignore
                    if use_tools and "tool_calls" in only_response:
                        # just one function anyway
                        function_call = only_response["tool_calls"][0][0]
                        function_name = function_call["name"]
                        function_args = json.loads(function_call["arguments"])
                        call_id = function_call["id"]
                        data_payload["messages"].append(
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": call_id,
                                        "type": "function",
                                        "function": {
                                            "name": function_name,
                                            "arguments": function_call["arguments"],
                                        },
                                    }
                                ],
                            }
                        )
                        data_payload["messages"].append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,  # A unique ID for the tool call
                                "content": "30",
                            }
                        )
                        response = query_llm.batch_requests(
                            metadata["endpoints"]["head"],
                            metadata["model_name"],
                            [data_payload],
                            app_name=app_name,
                            tool_choice=tool_choice,
                            tools=tools,
                            **kwargs,
                        )[0]
                        print(response)
                    return "error" not in response["response"]  # type: ignore[index]

    @property
    def app(self):
        """Manage applications."""

        from matrix.app_server.app_api import AppApi

        head = self.cluster.cluster_info()
        assert head, "head not started"
        assert self.cluster_id
        return AppApi(self.matrix_dir / self.cluster_id, head)

    @property
    def job(self):
        """Manage jobs."""

        from matrix.job.job_api import JobApi

        head = self.cluster.cluster_info()
        assert head, "head not started"
        assert self.cluster_id
        return JobApi(self.matrix_dir / self.cluster_id, head, self.app)


def main():
    fire.Fire(Cli)

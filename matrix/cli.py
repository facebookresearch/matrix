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

from matrix.app_server import deploy_app
from matrix.client import query_llm
from matrix.cluster.ray_cluster import RayCluster
from matrix.utils.os import run_subprocess


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
    ):
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
        self.cluster.start(
            add_workers,
            slurm,
            local,
            enable_grafana=enable_grafana,
            force_new_head=force_new_head,
        )

    def stop_cluster(self):
        """
        Shuts down the Ray cluster.

        This command gracefully terminates all processes in the Ray cluster,
        releasing resources back to the system or Slurm allocation.

        Returns:
            None
        """
        self.cluster.stop()

    def status(self, replica=False):
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
            print("head not started")
            return
        else:
            assert head.hostname

            print(
                f"ssh to head node:\nssh -L {head.dashboard_port}:localhost:{head.dashboard_port} -L {head.prometheus_port}:localhost:{head.prometheus_port} -L {head.grafana_port}:localhost:{head.grafana_port} {head.hostname}"  # type: ignore[union-attr]
            )
            cluster_info = deploy_app.convert_to_json_compatible(head)
            print("\nHead Info:")
            print(json.dumps(cluster_info, indent=2))

            print("\nRay status: --------")
            subprocess.run(
                ["ray", "status", "--address", f"{head.hostname}:{head.port}"]
            )
            print("\n\nServe status: --------")
            deploy_app.status(head, replica)

    def deploy_applications(
        self,
        action: str | deploy_app.Action = deploy_app.Action.REPLACE,
        applications: tp.Optional[tp.List[tp.Dict[str, tp.Union[str, int]]]] = None,
        yaml_config: tp.Optional[str] = None,
        block: bool = False,
    ):
        """
        Deploy applications on top of the Ray cluster.

        This method can deploy various applications such as vLLM serving,
        LLM proxies, or code execution services to the Ray cluster.

        Args:
            action (str | deploy_app.Action, optional): The deployment action to perform.
                Can be REPLACE, REMOVE, or ADD. Defaults to REPLACE.
            applications (List[Dict], optional): List of application configurations.
                Each dictionary should contain application specifications.
            yaml_config (str, optional): Path to a YAML file containing application
                configurations. Used as an alternative to the applications parameter.

        Returns:
            The deployed application names.
        """
        head = self.cluster.cluster_info()
        if not head:
            print("head not started")
            return
        else:
            assert head.hostname
            assert self.cluster_id
            return deploy_app.deploy(
                self.matrix_dir / self.cluster_id,
                head,
                action,
                applications,
                yaml_config,
                block=block,
            )

    def inference(self, app_name: str, output_jsonl: str, input_jsonls: str, **kwargs):
        """
        Run batch inference using a deployed application.

        This method processes input data through a deployed application and
        saves the results to the specified output file.

        Args:
            app_name (str): The name of the deployed application to use.
            output_jsonl (str): Path to save inference results in JSONL format.
            input_jsonls (str): Path to input data in JSONL format.
            **kwargs: Additional parameters for inference (e.g., temperature, max_tokens).

        Returns:
            None
        """
        head = self.cluster.cluster_info()
        if not head:
            print("head not started")
            return
        else:
            assert head.hostname
            assert self.cluster_id
            return deploy_app.inference(
                self.matrix_dir / self.cluster_id,
                head,
                app_name,
                output_jsonl,
                input_jsonls,
                **kwargs,
            )

    def get_app_metadata(
        self,
        app_name: str,
        endpoint_ttl_sec: int = 5,
        model_name: str | None = None,
        head_only: bool = False,
    ):
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
        head = self.cluster.cluster_info()
        if not head:
            print("head not started")
            return
        else:
            assert head.hostname
            assert self.cluster_id
            return deploy_app.get_app_metadata(
                self.matrix_dir / self.cluster_id,
                head,
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
            else:
                prompt = prompt or "What is 2+4=?"
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
                    if not use_chat:
                        data_payload = {"prompt": prompt}
                    response = query_llm.batch_requests(
                        metadata["endpoints"]["head"],
                        metadata["model_name"],
                        [data_payload],
                        app_name=app_name,
                        **kwargs,
                    )[0]
                    print(response)
                    return "error" not in response["response"]


def main():
    fire.Fire(Cli)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional

import aiohttp
from tqdm import tqdm

from matrix.utils.http import fetch_url, post_url

logger = logging.getLogger(__name__)

"""
How to set timeout?
1. in execute, we can block forever. this includes the wait time due to queueing
2. in acquire, when we preallocate containers, it is better to fail early when not enough containers. But the time needs to allow container image download. say 5 minutes.
"""


class ContainerClientError(Exception):
    """Custom exception for container client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class ContainerClient:
    """Client for interacting with the ContainerDeployment HTTP server."""

    def __init__(self, base_url: str):
        """
        Initialize the container client.

        Args:
            base_url: Base URL of the container deployment server (e.g., "http://localhost:8000")
        """
        self.base_url = base_url.rstrip("/")
        self.containers: list[str] = []

    def _parse_response(self, status: Optional[int], content: str) -> Dict:
        """Parse response content and handle errors."""
        if status is None:
            raise ContainerClientError(f"Request failed: {content}")

        if status >= 400:
            print(f"Error response from server: {content}")
            try:
                error_data = json.loads(content)
                detail = error_data.get("detail", content)
            except json.JSONDecodeError:
                detail = content
            raise ContainerClientError(f"HTTP {status}: {detail}", status_code=status)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            raise ContainerClientError(f"Invalid JSON response: {content}")

    async def acquire_container(
        self,
        image: str,
        executable: str = "apptainer",
        run_args: Optional[List[str]] = None,
        timeout: int = 300,
    ) -> str | None:
        """
        Acquire a container with the specified image.

        Args:
            image: Container image (e.g., "docker://ubuntu:22.04")
            executable: Container runtime executable (default: "apptainer")
            run_args: Additional runtime arguments (default: [])
            timeout: Timeout in seconds to wait for available container (default: 5.0)

        Returns:
            Container ID string

        Raises:
            ContainerClientError: If acquisition fails
        """
        if run_args is None:
            run_args = []

        payload = {
            "image": image,
            "executable": executable,
            "run_args": run_args,
            "timeout": timeout,
        }

        url = f"{self.base_url}/acquire"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(timeout)) as session:
            status, content = await post_url(session, url, payload)
            response = self._parse_response(status, content)
            container_id = response.get("container_id")
            if container_id:
                self.containers.append(container_id)

            return container_id

    async def release_container(
        self, container_id: str, timeout: int | None = None
    ) -> Dict:
        """
        Release a container by its ID.

        Args:
            container_id: ID of the container to release

        Returns:
            Response dictionary with status and container_id

        Raises:
            ContainerClientError: If release fails
        """
        payload = {"container_id": container_id}
        url = f"{self.base_url}/release"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(timeout)) as session:
            status, content = await post_url(session, url, payload)
            result = self._parse_response(status, content)
            if container_id in self.containers:
                self.containers.remove(container_id)
            return result

    async def release_all(self, timeout: int | None = None) -> Dict:
        """
        Release all containers owned by this client.

        Args:
            timeout: Timeout in seconds for the release operation (default: None)

        Returns:
            Response dictionary with status and released container IDs

        Raises:
            ContainerClientError: If release fails
        """
        url = f"{self.base_url}/release_all"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(timeout)) as session:
            status, content = await post_url(session, url, {})
            result = self._parse_response(status, content)
            self.containers.clear()
            return result

    async def execute(
        self,
        container_id: str,
        cmd: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        forward_env: Optional[List[str]] = None,
        timeout: int | None = None,  # wait indefinitely
    ) -> Dict:
        """
        Execute a command in the specified container.

        Args:
            container_id: ID of the container to execute command in
            cmd: Command to execute
            cwd: Working directory for the command (optional)
            env: Environment variables to set (optional)
            forward_env: List of environment variables to forward from host (optional)

        Returns:
            Execution result dictionary

        Raises:
            ContainerClientError: If execution fails
        """
        payload: Dict[str, Any] = {"container_id": container_id, "cmd": cmd}

        if cwd is not None:
            payload["cwd"] = cwd
        if env is not None:
            payload["env"] = env
        if forward_env is not None:
            payload["forward_env"] = forward_env
        payload["timeout"] = timeout

        url = f"{self.base_url}/execute"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(timeout)) as session:
            status, content = await post_url(session, url, payload)
            return self._parse_response(status, content)

    async def get_status(self) -> Dict:
        """
        Get the current status of the container deployment.

        Returns:
            Status information dictionary

        Raises:
            ContainerClientError: If status request fails
        """
        url = f"{self.base_url}/status"
        status, content = await fetch_url(url)
        return self._parse_response(status, content)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.containers:
            print(f"Releasing containers {self.containers}")
            tasks = [self.release_container(cid) for cid in self.containers]
            await asyncio.gather(*tasks, return_exceptions=False)
        return False  # re-raise exception if one happened


# Context manager for automatic container lifecycle management for one container
class ManagedContainer:
    """Context manager for automatic container acquisition and release."""

    def __init__(
        self,
        client: ContainerClient,
        image: str,
        executable: str = "apptainer",
        run_args: Optional[List[str]] = None,
        timeout: int = 300,
    ):
        self.client = client
        self.image = image
        self.executable = executable
        self.run_args = run_args or []
        self.timeout = timeout
        self.container_id: Optional[str] = None

    async def __aenter__(self) -> str | None:
        """Acquire container on entering context."""
        self.container_id = await self.client.acquire_container(
            image=self.image,
            executable=self.executable,
            run_args=self.run_args,
            timeout=self.timeout,
        )
        return self.container_id

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release container on exiting context."""
        if self.container_id:
            try:
                await self.client.release_container(self.container_id)
            except ContainerClientError:
                # Log error but don't raise - we're already exiting
                pass


if __name__ == "__main__":
    import sys

    from matrix.utils.os import batch_requests_async, run_async

    base_url = sys.argv[1]
    tags = ["22.04", "24.04", "25.04"]

    async def test_batch():
        async with ContainerClient(base_url) as client:
            containers = await batch_requests_async(
                client.acquire_container,
                [
                    {"executable": "apptainer", "image": f"docker://ubuntu:{tag}"}
                    for tag in tags
                ],
            )
            containers = [cid for cid in containers if not isinstance(cid, Exception)]
            await batch_requests_async(
                client.execute,
                [
                    {
                        "container_id": cid,
                        "cmd": "apt update && apt install -y lsb-release",
                    }
                    for cid in containers
                ],
            )
            outputs = await batch_requests_async(
                client.execute,
                [{"container_id": cid, "cmd": "lsb_release -r"} for cid in containers],
            )
            return outputs

    print(run_async(test_batch()))

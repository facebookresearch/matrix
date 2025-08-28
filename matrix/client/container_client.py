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

    async def _handle_response(
        self, status: Optional[int], content: str
    ) -> Dict[str, Any]:
        """Handle HTTP response and convert to standardized format."""
        if status is None:
            # Network or connection error
            return {"error": content}

        try:
            # Try to parse JSON response
            response_data = json.loads(content)

            # Check if it's an HTTP error status
            if status >= 400:
                if isinstance(response_data, dict) and "detail" in response_data:
                    return {"error": response_data["detail"]}
                else:
                    return {"error": f"HTTP {status}: {content}"}

            return response_data

        except json.JSONDecodeError:
            if status >= 400:
                return {"error": f"HTTP {status}: {content}"}
            else:
                return {"error": f"Invalid JSON response: {content}"}

    async def acquire_container(
        self,
        image: str,
        executable: str = "apptainer",
        run_args: Optional[List[str]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Acquire a new container.

        Args:
            image: Container image (e.g., "docker://ubuntu:22.04")
            executable: Container runtime executable (default: "apptainer")
            run_args: Additional arguments for container run (default: [])
            timeout: Timeout for container acquisition

        Returns:
            Dict with either {"container_id": "..."} or {"error": "..."}
        """
        if run_args is None:
            run_args = []

        payload = {
            "image": image,
            "executable": executable,
            "run_args": run_args,
            "timeout": timeout,
        }

        session_timeout = aiohttp.ClientTimeout(total=timeout + 5) if timeout else None
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            status, content = await post_url(
                session, f"{self.base_url}/acquire", payload
            )
            return await self._handle_response(status, content)

    async def release_container(self, container_id: str) -> Dict[str, Any]:
        """
        Release a container.

        Args:
            container_id: ID of the container to release

        Returns:
            Dict with either {"container_id": "..."} or {"error": "..."}
        """
        payload = {"container_id": container_id}

        async with aiohttp.ClientSession() as session:
            status, content = await post_url(
                session, f"{self.base_url}/release", payload
            )
            return await self._handle_response(status, content)

    async def execute(
        self,
        container_id: str,
        cmd: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        forward_env: Optional[List[str]] = None,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        Execute a command in a container.

        Args:
            container_id: ID of the container
            cmd: Command to execute
            cwd: Working directory for command execution
            env: Environment variables to set
            forward_env: Environment variables to forward from host
            timeout: Command timeout in seconds (default: 30)

        Returns:
            Dict with either {"returncode": int, "output": str} or {"error": "..."}
        """
        payload = {"container_id": container_id, "cmd": cmd, "timeout": timeout}
        if cwd is not None:
            payload["cwd"] = cwd
        if env is not None:
            payload["env"] = env
        if forward_env is not None:
            payload["forward_env"] = forward_env

        session_timeout = aiohttp.ClientTimeout(total=timeout + 5) if timeout else None
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            status, content = await post_url(
                session, f"{self.base_url}/execute", payload
            )
            return await self._handle_response(status, content)

    async def get_status(self) -> Dict[str, Any]:
        """
        Get status of all containers and actors.

        Returns:
            Dict with either {"actors": {...}, "containers": {...}} or {"error": "..."}
        """
        status, content = await fetch_url(f"{self.base_url}/status")
        return await self._handle_response(status, content)

    async def release_all_containers(self) -> Dict[str, Any]:
        """
        Release all containers.

        Returns:
            Dict with either {"container_ids": []} or {"error": "..."}
        """
        async with aiohttp.ClientSession() as session:
            status, content = await post_url(
                session, f"{self.base_url}/release_all", {}
            )
            return await self._handle_response(status, content)

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

    async def __aenter__(self) -> "ManagedContainer":
        """Acquire container on entering context."""
        result = await self.client.acquire_container(
            image=self.image,
            executable=self.executable,
            run_args=self.run_args,
            timeout=self.timeout,
        )
        if "error" in result:
            raise Exception(f"Failed to acquire container: {result['error']}")
        self.container_id = result["container_id"]
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release container on exiting context."""
        if self.container_id:
            await self.client.release_container(self.container_id)
            self.container_id = None

    async def execute(
        self,
        cmd: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        forward_env: Optional[List[str]] = None,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        assert self.container_id is not None, "Container not acquired yet"
        return await self.client.execute(
            self.container_id, cmd, cwd, env, forward_env, timeout
        )


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
            print(containers)
            containers = [
                cid["container_id"] for cid in containers if "error" not in cid
            ]
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

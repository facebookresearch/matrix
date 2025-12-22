# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union

import aiohttp

from matrix.utils.http import post_url

logger = logging.getLogger(__name__)


class ContainerClient:
    """Picklable, scalable client for interacting with the ContainerDeployment HTTP server.

    This client is safe to use across different event loops and threads.
    """

    def __init__(
        self,
        base_url: str,
        max_connections: int = 500,
        max_connections_per_host: int = 0,
        timeout: Optional[float] = 60.0,
    ):
        """
        Initialize the container client.

        Args:
            base_url: Base URL of the container deployment server (e.g., "http://localhost:8000")
            max_connections: Maximum number of concurrent connections (default: 100)
            max_connections_per_host: Max connections per host, 0 for unlimited (default: 0)
            timeout: Default timeout in seconds for all requests (default: 60.0)
        """
        self.base_url = base_url.rstrip("/")

        # Store configuration (these are picklable)
        self._max_connections = max_connections
        self._max_connections_per_host = max_connections_per_host
        self._timeout = timeout

        # Per-loop session storage - each event loop gets its own session
        self._sessions: Dict[asyncio.AbstractEventLoop, aiohttp.ClientSession] = {}
        self._connectors: Dict[asyncio.AbstractEventLoop, aiohttp.TCPConnector] = {}
        self._locks: Dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}

    def __getstate__(self):
        """Custom pickle serialization - exclude non-picklable objects."""
        state = self.__dict__.copy()
        # Remove non-picklable session and connector dictionaries
        state["_sessions"] = {}
        state["_connectors"] = {}
        state["_locks"] = {}
        return state

    def __setstate__(self, state):
        """Custom pickle deserialization - restore state."""
        self.__dict__.update(state)
        # Recreate empty dictionaries after unpickling
        self._sessions = {}
        self._connectors = {}
        self._locks = {}

    def _get_lock(self) -> asyncio.Lock:
        """Get or create a lock for the current event loop."""
        loop = asyncio.get_event_loop()
        if loop not in self._locks:
            self._locks[loop] = asyncio.Lock()
        return self._locks[loop]

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure the session is created for the current event loop (thread-safe)."""
        loop = asyncio.get_event_loop()

        # Check if we have a valid session for this loop
        if loop in self._sessions and not self._sessions[loop].closed:
            return self._sessions[loop]

        # Need to create a new session for this loop
        lock = self._get_lock()
        async with lock:
            # Double-check after acquiring lock
            if loop in self._sessions and not self._sessions[loop].closed:
                return self._sessions[loop]

            # Clean up old session if it exists
            if loop in self._sessions:
                try:
                    await self._sessions[loop].close()
                except:
                    pass
            if loop in self._connectors:
                try:
                    await self._connectors[loop].close()
                except:
                    pass

            # Create new connector and session for this loop
            connector = aiohttp.TCPConnector(
                limit=self._max_connections,
                limit_per_host=self._max_connections_per_host,
                ttl_dns_cache=300,
            )

            timeout = aiohttp.ClientTimeout(total=self._timeout)
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                connector_owner=False,
            )

            self._connectors[loop] = connector
            self._sessions[loop] = session

            return session

    async def close(self):
        """Close all client sessions and connectors."""
        # Get all loops to close
        loops_to_close = list(self._sessions.keys())

        for loop in loops_to_close:
            if loop in self._sessions:
                try:
                    session = self._sessions[loop]
                    if not session.closed:
                        await session.close()
                except:
                    pass

            if loop in self._connectors:
                try:
                    connector = self._connectors[loop]
                    if not connector.closed:
                        await connector.close()
                except:
                    pass

        # Clear dictionaries
        self._sessions.clear()
        self._connectors.clear()

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
        start_script_args: Optional[List[str]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Acquire a new container.

        Args:
            image: Container image (e.g., "docker://ubuntu:22.04")
            executable: Container runtime executable (default: "apptainer")
            run_args: Additional arguments for container run (default: [])
            start_script_args: Arguments to pass to the container start script (default: [])
            timeout: Timeout for container acquisition

        Returns:
            Dict with either {"container_id": "..."} or {"error": "..."}
        """
        session = await self._ensure_session()

        payload = {
            "image": image,
            "executable": executable,
            "run_args": run_args,
            "start_script_args": start_script_args,
            "timeout": timeout,
        }

        try:
            status, content = await post_url(
                session, f"{self.base_url}/acquire", payload
            )
            return await self._handle_response(status, content)
        except asyncio.TimeoutError:
            return {
                "error": f"Request timed out after {timeout + 5 if timeout else self._timeout} seconds"
            }
        except aiohttp.ClientError as e:
            return {"error": f"Client error: {repr(e)}"}

    async def release_container(self, container_id: str) -> Dict[str, Any]:
        """
        Release a container.

        Args:
            container_id: ID of the container to release

        Returns:
            Dict with either {"container_id": "..."} or {"error": "..."}
        """
        session = await self._ensure_session()

        payload = {"container_id": container_id}

        try:
            status, content = await post_url(
                session, f"{self.base_url}/release", payload
            )
            return await self._handle_response(status, content)
        except asyncio.TimeoutError:
            return {"error": f"Request timed out after {self._timeout} seconds"}
        except aiohttp.ClientError as e:
            return {"error": f"Client error: {repr(e)}"}

    async def execute(
        self,
        container_id: str,
        cmd: Union[List[str], str],
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
        session = await self._ensure_session()

        payload = {"container_id": container_id, "cmd": cmd, "timeout": timeout}
        if cwd is not None:
            payload["cwd"] = cwd
        if env is not None:
            payload["env"] = env  # type: ignore[assignment]
        if forward_env is not None:
            payload["forward_env"] = forward_env

        try:
            status, content = await post_url(
                session, f"{self.base_url}/execute", payload
            )
            return await self._handle_response(status, content)
        except asyncio.TimeoutError:
            return {"error": f"Request timed out after {timeout + 5} seconds"}
        except aiohttp.ClientError as e:
            return {"error": f"Client error: {repr(e)}"}

    async def get_status(self) -> Dict[str, Any]:
        """
        Get status of all containers and actors.

        Returns:
            Dict with either {"actors": {...}, "containers": {...}} or {"error": "..."}
        """
        session = await self._ensure_session()

        try:
            async with session.get(f"{self.base_url}/status") as response:
                status = response.status
                content = await response.text()
                return await self._handle_response(status, content)
        except asyncio.TimeoutError:
            return {"error": f"Request timed out after {self._timeout} seconds"}
        except aiohttp.ClientError as e:
            return {"error": f"Client error: {repr(e)}"}

    async def release_all_containers(self) -> Dict[str, Any]:
        """
        Release all containers.

        Returns:
            Dict with either {"container_ids": []} or {"error": "..."}
        """
        session = await self._ensure_session()

        try:
            status, content = await post_url(
                session, f"{self.base_url}/release_all", {}
            )
            return await self._handle_response(status, content)
        except asyncio.TimeoutError:
            return {"error": f"Request timed out after {self._timeout} seconds"}
        except aiohttp.ClientError as e:
            return {"error": f"Client error: {repr(e)}"}

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        try:
            await self.release_all_containers()
        finally:
            await self.close()
        return False  # re-raise exception if one happened


# Context manager for automatic container lifecycle management for one container
class ManagedContainer:
    """Context manager for automatic container acquisition and release."""

    def __init__(
        self,
        base_url: str,
        image: str,
        executable: str = "apptainer",
        run_args: Optional[List[str]] = None,
        start_script_args: Optional[List[str]] = None,
        timeout: int = 300,
    ):
        self.base_url = base_url
        self.client = ContainerClient(base_url)
        self.image = image
        self.executable = executable
        self.run_args = run_args
        self.start_script_args = start_script_args
        self.timeout = timeout
        self.container_id: Optional[str] = None

    async def __aenter__(self) -> "ManagedContainer":
        """Acquire container on entering context."""
        await self.client.__aenter__()
        result = await self.client.acquire_container(
            image=self.image,
            executable=self.executable,
            run_args=self.run_args,
            start_script_args=self.start_script_args,
            timeout=self.timeout,
        )
        if "error" in result:
            raise Exception(f"Failed to acquire container: {result['error']}")
        self.container_id = result["container_id"]
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release container on exiting context."""
        await self.client.__aexit__(exc_type, exc_val, exc_tb)
        self.container_id = None

    async def execute(
        self,
        cmd: Union[List[str], str],
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

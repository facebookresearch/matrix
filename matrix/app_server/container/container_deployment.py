# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import os
import random
import shlex
import subprocess
import uuid
from typing import Any, Dict, Optional

import ray
from fastapi import FastAPI, HTTPException
from ray import serve

"""
ContainerDeployment has several replicas controlled by user.
each replica has num_container ContainerActor, created when replica deploy.
each ContainerActor has one container. container won't start until acquire, container removed until release. After container release, another container can start.
"""

logger = logging.getLogger("container")


# ----------------------------
# ContainerRegistry (detached)
# ----------------------------
@ray.remote
class ContainerRegistry:
    def __init__(self):
        # container_id (hex) -> {"handle": ActorHandle, "owner": replica_id}
        # container_is is owned by the replica that created it, when replica die, container will die and should be removed
        self.containers: Dict[str, Dict[str, Any]] = {}
        # container_id -> replica_id (hex)
        # container_id is serving traffic from replica_id, when replica die, these containers become free
        self.assignment: Dict[str, str] = {}

    def _actor_id_from_handle(self, handle):
        # ActorHandle._actor_id is internal, but works; store hex.
        return handle._actor_id.hex()

    def register_actor(self, owner_id: str, handle, container_id: str):
        self.containers[container_id] = {"handle": handle, "owner": owner_id}
        return container_id

    def get_container_handle(self, container_id: str):
        """
        Returns (actor_handle, actor_id_hex) or (None, None)
        Cleans up if actor is dead (lazy).
        """
        info = self.containers.get(container_id)
        if not info:
            return None
        # check if the container is assigned
        if container_id not in self.assignment:
            return None
        return info["handle"]

    def acquire(self, replica_id: str) -> Optional[tuple[str, ray.actor.ActorHandle]]:
        """
        Return an idle actor handle and its id hex. Does NOT remove it from pool.
        We'll consider any actor that does not have a container assigned as idle.
        Cleans dead actors lazily.
        """
        # Build set of busy actor ids
        busy = set(self.assignment.keys())
        # iterate available actors, prefer owner if provided
        available = [
            (cid, info) for cid, info in self.containers.items() if cid not in busy
        ]
        if available:
            # randomly select one
            cid, info = random.choice(available)
            self.assignment[cid] = replica_id
            return cid, info["handle"]
        else:
            return None, None

    def release(self, container_id: str):
        self.assignment.pop(container_id, None)
        return True

    def list_actors(self):
        return {
            "containers": {cid: info["owner"] for cid, info in self.containers.items()},
            "assignment": self.assignment,
        }

    def cleanup_dead_container(self, container_id: str):
        # Caller requested explicit removal
        self.containers.pop(container_id, None)
        self.assignment.pop(container_id, None)

    def cleanup_replica(self, replica_id: str):
        """
        Cleanup all actors owned by this replica.
        """
        logger.info(f"Cleaning up dead replica {replica_id}")
        to_remove = [
            cid for cid, info in self.containers.items() if info["owner"] == replica_id
        ]
        for cid in to_remove:
            self.containers.pop(cid, None)
        to_unassign = [
            cid for cid, owner in self.assignment.items() if owner == replica_id
        ]
        for cid in to_unassign:
            self.assignment.pop(cid, None)


# ----------------------------
# Generic ContainerActor base
# ----------------------------
@ray.remote(num_cpus=1)
class ContainerActor:
    def __init__(self):
        self.container_id = f"container-{uuid.uuid4().hex[:8]}"
        self.process = None

    def get_id(self):
        return self.container_id

    def start_container(self, **config):
        """Start the Apptainer instance (persistent container)."""
        self.config = config
        cmd = [self.config["executable"], "instance", "start", "--fakeroot"]
        cmd.append("--writable-tmpfs")
        cmd.extend(self.config["run_args"])
        cmd.extend([self.config["image"], self.container_id])

        print(f"Starting instance with command: {shlex.join(cmd)}")
        self.process = subprocess.run(
            cmd, check=True, timeout=3600 * 2
        )  # 2 hours timeout

    def execute(
        self,
        command: str,
        cwd: str = "",
        env: dict[str, str] = None,
        forward_env: list[str] = None,
        timeout_secs: int | None = None,
    ) -> dict[str, Any]:
        """Run a command inside the running instance."""
        if self.process is None:
            raise RuntimeError(
                "Container instance not started. Call start_container() first."
            )

        work_dir = cwd or self.config.get("cwd")

        cmd = [self.config["executable"], "exec"]
        if work_dir and work_dir != "/":
            cmd.extend(["--pwd", work_dir])

        for key in forward_env or []:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["--env", f"{key}={value}"])
        for key, value in (env or {}).items():
            cmd.extend(["--env", f"{key}={value}"])

        cmd.append(f"instance://{self.container_id}")
        cmd.extend(["bash", "-lc", command])

        result = subprocess.run(
            cmd,
            text=True,
            timeout=timeout_secs,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}

    def cleanup(self):
        """Stop the Apptainer instance."""
        if self.process is not None:
            print(f"Stopping instance {self.container_id}")
            stop_cmd = [
                self.config["executable"],
                "instance",
                "stop",
                self.container_id,
            ]
            subprocess.Popen(stop_cmd)
            self.process = None

    def __del__(self):
        self.cleanup()


# ----------------------------
# Serve deployment that creates local actors and registers them
# ----------------------------
app = FastAPI()


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 8,
        "target_ongoing_requests": 32,
    },
    max_ongoing_requests=32,
)
@serve.ingress(app)
class ContainerDeployment:
    def __init__(self, num_containers: int = 32):
        # identify this replica
        # keep this simple: use a uuid per replica
        self.replica_id = f"replica-{uuid.uuid4().hex[:8]}"
        self.num_containers = num_containers

        # Get or create the detached global registry by name
        try:
            self.registry = ray.get_actor(
                "system.container_registry", namespace="matrix"
            )
        except ValueError:
            # todo, in case of race, one will success, catch and get_actor again
            try:
                self.registry = ContainerRegistry.options(
                    name="system.container_registry",
                    namespace="matrix",
                    lifetime="detached",
                ).remote()
            except Exception as e:
                logger.error(f"Failed to create container registry: {e}")
                self.registry = ray.get_actor(
                    "system.container_registry", namespace="matrix"
                )
        # create local non-detached actors and register them
        self.local_actor_ids = []  # actor ids hex owned by this replica
        for _ in range(self.num_containers):
            c_handle = ContainerActor.remote()
            c_id = ray.get(c_handle.get_id.remote())
            ray.get(
                self.registry.register_actor.remote(self.replica_id, c_handle, c_id)
            )
            self.local_actor_ids.append(c_id)

    async def _ray_get(self, ref):
        # helper to await ray.get without blocking the async loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, ray.get, ref)

    @app.post("/acquire")
    async def acquire_container(self, payload: Dict):
        """
        payload: {"timeout_s": 5, "executable": "apptainer", "image": "docker://ubuntu:22.04", "run_args": []}
        returns {"container_id": ...}
        """
        image = payload.get("image")
        if not image:
            raise HTTPException(status_code=400, detail="image required")
        executable = payload.get("executable", "apptainer")
        run_args = payload.get("run_args", [])

        container_id = payload.get("container_id", None)
        assert container_id is None, "container_id unexpected"
        timeout_s = float(payload.get("timeout_s", 5.0))

        start = asyncio.get_event_loop().time()
        while True:
            container_id, handle = await self._ray_get(
                self.registry.acquire.remote(self.replica_id)
            )
            if handle is not None:
                try:
                    await self._ray_get(
                        handle.start_container.remote(
                            executable=executable,
                            image=image,
                            run_args=run_args,
                        )
                    )
                    return {"container_id": container_id}
                except Exception as e:
                    # actor probably died or failed - do a cleanup of that actor in registry
                    await self._ray_get(handle.cleanup.remote())
                    await self._ray_get(self.registry.release.remote(container_id))

                    raise HTTPException(
                        status_code=500, detail=f"actor execution failed: {e}"
                    )

            # none available
            if asyncio.get_event_loop().time() - start > timeout_s:
                raise HTTPException(
                    status_code=503, detail="No available containers, wait then retry."
                )
            await asyncio.sleep(1)

    @app.post("/release")
    async def release_container(self, payload: Dict):
        """
        payload: {"container_id": "..."}
        """
        container_id = payload.get("container_id")
        if not container_id:
            raise HTTPException(status_code=400, detail="container_id required")

        # lookup actor for container
        handle = await self._ray_get(
            self.registry.get_container_handle.remote(container_id)
        )
        if handle is None:
            raise HTTPException(
                status_code=404, detail=f"bad container id {container_id}"
            )
        await self._ray_get(handle.cleanup.remote())

        await self._ray_get(self.registry.release.remote(container_id))
        return {"status": "ok", "container_id": container_id}

    @app.post("/execute")
    async def execute(self, payload: Dict):
        """
        payload: {"container_id": "...", "cmd": "..."}
        """
        container_id = payload.get("container_id")
        cmd = payload.get("cmd")
        if not container_id or not cmd:
            raise HTTPException(status_code=400, detail="container_id and cmd required")
        cwd = payload.get("cwd")
        env = payload.get("env")
        forward_env = payload.get("forward_env")

        # lookup actor for container
        handle = await self._ray_get(
            self.registry.get_container_handle.remote(container_id)
        )
        if handle is None:
            raise HTTPException(
                status_code=404, detail=f"bad container id {container_id}"
            )

        # call the actor.execute remotely; await result
        try:
            return await self._ray_get(
                handle.execute.remote(cmd, cwd=cwd, env=env, forward_env=forward_env)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"actor execution failed: {e}")

    @app.get("/status")
    async def status(self):
        info = await self._ray_get(self.registry.list_actors.remote())
        return info

    def __del__(self):
        """Clean up this replica when it's destroyed"""
        try:
            # This might not work reliably in all shutdown scenarios
            ray.get(self.registry.cleanup_replica.remote(self.replica_id))
        except Exception:
            pass


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments."""  # noqa: E501
    pg_resources = []
    pg_resources.append({"CPU": 2})  # for the deployment replica

    # We use the "STRICT_PACK" strategy below to ensure all vLLM actors are placed on
    # the same Ray node.
    return ContainerDeployment.options(  # type: ignore[union-attr]
        placement_group_bundles=pg_resources,
        placement_group_strategy="STRICT_PACK",
    ).bind(**cli_args)

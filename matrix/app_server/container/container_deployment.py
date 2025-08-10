# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import os
import random
import shlex
import subprocess
import uuid
from typing import Any, Dict, Optional

import ray
from fastapi import FastAPI, HTTPException
from ray import serve


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

    def register_actor(self, owner_id: str, handle):
        aid = self._actor_id_from_handle(handle)
        self.containers[aid] = {"handle": handle, "owner": owner_id}
        return aid

    def get_container_handle(self, container_id: str):
        """
        Returns (actor_handle, actor_id_hex) or (None, None)
        Cleans up if actor is dead (lazy).
        """
        info = self.containers.get(container_id)
        if not info:
            return None
        return info["handle"]

    def acquire(self, owner_id: Optional[str] = None):
        """
        Return an idle actor handle and its id hex. Does NOT remove it from pool.
        We'll consider any actor that does not have a container assigned as idle.
        Cleans dead actors lazily.
        """
        # Build set of busy actor ids
        busy = set(self.assignment.key())
        # iterate available actors, prefer owner if provided
        available = [
            (cid, info) for cid, info in self.containers.items() if cid not in busy
        ]
        if available:
            # randomly select one
            cid, info = random.choice(available)
            return cid, info["handle"]
        else:
            return None

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
        to_remove = [
            cid for cid, info in self.containers.items() if info["owner"] == replica_id
        ]
        for cid in to_remove:
            self.containers.pop(cid)
        to_unassign = [
            cid for cid, owner in self.assignment.items() if owner == replica_id
        ]
        for cid in to_unassign:
            self.assignment.pop(cid, None)


# ----------------------------
# Generic ContainerActor base
# ----------------------------
@ray.remote(num_gpus=2)
class ContainerActor:
    def __init__(self):
        self.instance_name = f"container-{uuid.uuid4().hex[:8]}"

    def start_container(self, **config):
        """Start the Apptainer instance (persistent container)."""
        self.config = config
        cmd = [
            self.config["executable"],
            "instance",
            "start",
        ]
        cmd.append("--writable-tmpfs")
        cmd.extend(self.config["run_args"])
        cmd.extend([self.config["image"], self.instance_name])

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
    ) -> dict[str, Any]:
        """Run a command inside the running instance."""
        work_dir = cwd or self.config.get("cwd")

        cmd = [self.config["executable"], "exec"]
        if work_dir and work_dir != "/":
            cmd.extend(["--pwd", work_dir])

        for key in forward_env or []:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["--env", f"{key}={value}"])
        for key, value in (env or {}).items():
            cmd.extend(["--env", f"{key}={value}"])

        cmd.append(f"instance://{self.instance_name}")
        cmd.extend(["bash", "-lc", command])

        result = subprocess.run(
            cmd,
            text=True,
            timeout=self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}

    def cleanup(self):
        """Stop the Apptainer instance."""
        if getattr(self, "instance_name", None) is not None:
            print(f"Stopping instance {self.instance_name}")
            stop_cmd = [self.config.executable, "instance", "stop", self.instance_name]
            subprocess.Popen(stop_cmd)

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
    def __init__(self, per_replica_containers: int = 32):
        # identify this replica
        # keep this simple: use a uuid per replica
        self.replica_id = str(uuid.uuid4())
        self.per_replica_containers = per_replica_containers

        # Get or create the detached global registry by name
        try:
            self.registry = ray.get_actor(
                "system.container_registry", namespace="matrix"
            )
        except ValueError:
            # todo, in case of race, one will success, catch and get_actor again
            self.registry = ContainerRegistry.options(
                name="system.container_registry",
                namespace="matrix",
                lifetime="detached",
            ).remote()

        # create local non-detached actors and register them
        self.local_actor_ids = []  # actor ids hex owned by this replica
        for _ in range(self.per_replica_containers):
            a = ContainerActor.remote()
            aid = ray.get(self.registry.register_actor.remote(self.replica_id, a))
            self.local_actor_ids.append(aid)

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
        timeout_s = float(payload.get("timeout_s", 30.0))

        start = asyncio.get_event_loop().time()
        while True:
            container_id, handle = await self._ray_get(
                self.registry.acquire.remote(None)
            )
            if handle is not None:
                handle.start_container.remote(
                    executable=executable,
                    image=image,
                    run_args=run_args,
                )
                return {"container_id": container_id}
            # none available
            if asyncio.get_event_loop().time() - start > timeout_s:
                raise HTTPException(
                    status_code=503, detail="No available containers (timeout)"
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
                status_code=404, detail="no actor assigned to container"
            )
        handle.cleanup.remote()

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
                status_code=404, detail="no actor assigned to container"
            )

        # call the actor.execute remotely; await result
        try:
            ref = handle.execute.remote(cmd, cwd=cwd, env=env, forward_env=forward_env)
            res = await self._ray_get(ref)
            return res
        except Exception as e:
            # actor probably died or failed - do a cleanup of that actor in registry
            try:
                await self._ray_get(
                    self.registry.cleanup_dead_container.remote(container_id)
                )
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"actor execution failed: {e}")

    @app.get("/status")
    async def status(self):
        info = await self._ray_get(self.registry.list_actors.remote())
        return {
            "replica_id": self.replica_id,
            "registry_actors": info,
            "local_actor_ids": self.local_actor_ids,
        }

    def __del__(self):
        """Clean up this replica when it's destroyed"""
        try:
            # This might not work reliably in all shutdown scenarios
            ray.get(self.registry.cleanup_replica.remote(self.replica_id))
        except Exception:
            pass


# bind and deploy
ContainerDeployment.options(name="container_deployment").deploy(
    per_replica_containers=2
)

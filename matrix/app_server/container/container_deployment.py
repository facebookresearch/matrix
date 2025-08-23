# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import atexit
import logging
import os
import random
import shlex
import subprocess
import threading
import time
import uuid
from functools import partial
from typing import Any, Dict, Optional

import ray
from fastapi import FastAPI, HTTPException
from ray import serve
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from matrix.utils.ray import ACTOR_NAME_SPACE, get_matrix_actors, get_ray_head_node

"""
ContainerDeployment has several replicas controlled by user.
each replica has num_containers_per_replica ContainerActor, created when replica deploy.
each ContainerActor has one container. container won't start until acquire, container is removed when release.
"""


# ----------------------------
# ContainerRegistry (detached)
# ----------------------------
@ray.remote(num_cpus=0)
class ContainerRegistry:
    name = "system.container_registry"

    def __init__(self):
        # actor_id (hex) -> {"handle": ActorHandle, "owner": replica_id}
        # actor is owned by the replica that created it, when replica die, actor will die and should be removed
        self.actors: Dict[str, Dict[str, Any]] = {}
        # container_id -> actor_id (hex)
        self.containers: Dict[str, str] = {}

    def register_actor(self, owner_id: str, handle, actor_id: str):
        print(f"register {actor_id} {owner_id}")  # note: logger.info does not print anything
        self.actors[actor_id] = {"handle": handle, "owner": owner_id}
        return actor_id

    def get_container_handle(self, container_id: str):
        """
        Returns (actor_handle, actor_id_hex) or (None, None)
        Cleans up if actor is dead (lazy).
        """
        actor_id = self.containers.get(container_id)
        if not actor_id:
            return None
        info = self.actors.get(actor_id)
        if not info:
            return None
        return info["handle"]

    def acquire(
        self, container_id: str
    ) -> tuple[str | None, ray.actor.ActorHandle | None]:
        """
        Return an idle actor id and handle.
        """
        # Build set of busy actor ids
        busy = set(self.containers.values())
        # iterate available actors
        available = [
            (aid, info) for aid, info in self.actors.items() if aid not in busy
        ]
        if available:
            # randomly select one
            aid, info = random.choice(available)
            self.containers[container_id] = aid
            print(f"acquire {container_id} {aid}")
            return aid, info["handle"]
        else:
            return None, None

    def release(self, container_id: str):
        print(f"release {container_id}")
        self.containers.pop(container_id, None)
        return True

    def list_actors(self):
        return {
            "actors": {aid: info["owner"] for aid, info in self.actors.items()},
            "containers": self.containers,
        }

    def release_all_containers(self):
        print(f"Release all containers")
        self.containers.clear()

    def cleanup_replica(self, replica_id: str):
        """
        Cleanup all actors owned by this replica.
        """
        print(f"Cleaning up dead replica {replica_id}")
        to_remove = [
            aid for aid, info in self.actors.items() if info["owner"] == replica_id
        ]
        for aid in to_remove:
            self.actors.pop(aid, None)
        to_unassign = [cid for cid, aid in self.containers.items() if aid in to_remove]
        for cid in to_unassign:
            self.containers.pop(cid, None)


# ----------------------------
# Generic ContainerActor base
# ----------------------------
@ray.remote(num_cpus=1)
class ContainerActor:
    def __init__(self):
        self.actor_id = f"actor-{uuid.uuid4().hex[:8]}"
        self.config = None
        atexit.register(self.cleanup)

    def get_id(self):
        return self.actor_id

    def start_container(self, **config):
        """Start the Apptainer instance (persistent container). May raise subprocess.CalledProcessError if failed."""
        self.config = config
        cmd = [self.config["executable"], "instance", "start", "--fakeroot"]
        cmd.append("--writable-tmpfs")
        cmd.extend(self.config["run_args"])
        cmd.extend([self.config["image"], self.config["container_id"]])

        print(f"Starting instance with command: {shlex.join(cmd)}")
        # Start the instance (blocking call, exits when daemon is launched)
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    def execute(
        self,
        command: str,
        cwd: str = "",
        env: dict[str, str] = None,
        forward_env: list[str] = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Run a command inside the running instance."""
        if self.config is None:
            raise RuntimeError(
                "Container instance not started. Call start_container() first."
            )

        container_id = self.config["container_id"]
        work_dir = cwd or self.config.get("cwd")

        cmd = [self.config["executable"], "exec"]
        if work_dir and work_dir != "/":
            cmd.extend(["--pwd", work_dir])

        for key in forward_env or []:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["--env", f"{key}={value}"])
        for key, value in (env or {}).items():
            cmd.extend(["--env", f"{key}={value}"])

        cmd.append(f"instance://{container_id}")
        cmd.extend(["bash", "-lc", command])

        result = subprocess.run(
            cmd,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}

    def cleanup(self):
        """Stop the Apptainer instance."""
        if self.config is not None:
            container_id = self.config["container_id"]
            print(f"Stopping instance {container_id}")
            stop_cmd = [
                self.config["executable"],
                "instance",
                "stop",
                container_id,
            ]
            proc = subprocess.Popen(stop_cmd)
            proc.wait()
            self.config = None

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
    def __init__(self, num_containers_per_replica: int = 32):
        self.registry = ray.get_actor(
            ContainerRegistry.name, namespace=ACTOR_NAME_SPACE
        )
        # identify this replica
        # keep this simple: use a uuid per replica
        self.replica_id = f"replica-{uuid.uuid4().hex[:8]}"
        self.num_containers_per_replica = num_containers_per_replica

        # create local non-detached actors and register them
        self.local_actors = []  # actor ids hex owned by this replica
        # Cancellation event
        self._stop_event = threading.Event()

        # Start background thread
        self._thread = threading.Thread(target=self._launch_actors, daemon=True)
        self._thread.start()

    def _launch_actors(self):
        """Create actors with infinite retry until stopped."""
        for _ in range(self.num_containers_per_replica):
            if self._stop_event.is_set():
                break
            actor_handle = ContainerActor.remote()
            while not self._stop_event.is_set():
                try:
                    # Use non-blocking wait with timeout
                    ready, _ = ray.wait(
                        [actor_handle.get_id.remote()],
                        timeout=2,  # short timeout for interruptibility
                    )

                    if ready:
                        actor_id = ray.get(ready[0])
                        # Register actor
                        ray.get(
                            self.registry.register_actor.remote(
                                self.replica_id, actor_handle, actor_id
                            )
                        )
                        self.local_actors.append(actor_handle)
                        break  # move to next actor
                except Exception as e:
                    # Could log or add exponential backoff
                    time.sleep(1)

    async def _ray_get(self, ref, get_fn=ray.get):
        # helper to await ray.get without blocking the async loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, get_fn, ref)

    @app.post("/acquire")
    async def acquire_container(self, payload: Dict):
        """
        payload: {"timeout": 5, "executable": "apptainer", "image": "docker://ubuntu:22.04", "run_args": []}
        returns {"container_id": ...}
        """
        image = payload.get("image")
        if not image:
            raise HTTPException(status_code=400, detail="image required")
        executable = payload.get("executable", "apptainer")
        run_args = payload.get("run_args", [])

        container_id = payload.get("container_id", None)
        assert container_id is None, "container_id unexpected"
        container_id = f"container-{uuid.uuid4().hex[:8]}"
        timeout = float(payload.get("timeout", 5.0))

        start = asyncio.get_event_loop().time()
        while True:
            _actor_id, handle = await self._ray_get(
                self.registry.acquire.remote(container_id)
            )
            if handle is not None:
                try:
                    await self._ray_get(
                        handle.start_container.remote(
                            executable=executable,
                            image=image,
                            run_args=run_args,
                            container_id=container_id,
                        )
                    )
                    return {"container_id": container_id}
                except Exception as e:
                    # actor probably died or failed - do a cleanup of that actor in registry
                    try:
                        await self._ray_get(handle.cleanup.remote())
                        await self._ray_get(self.registry.release.remote(container_id))
                    except Exception:
                        pass

                    raise HTTPException(
                        status_code=500, detail=f"Failed to start_container: {e}"
                    )

            # none available
            if asyncio.get_event_loop().time() - start > timeout:
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
        try:
            await self._ray_get(handle.cleanup.remote())
        except Exception:
            pass
        try:
            await self._ray_get(self.registry.release.remote(container_id))
        except Exception as e:
            return {"status": "failed", "container_id": container_id, "error": repr(e)}
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

    @app.post("/release_all")
    async def release_all_containers(self, payload: Dict):
        """
        remove all live containers
        """
        actors_containers = await self._ray_get(self.registry.list_containers.remote())
        actors = actors_containers["actors"]
        containers = actors_containers["containers"]
        handles = [
            actors[aid]["handle"] for aid in containers.values() if aid in actors
        ]
        try:
            await self._ray_get(
                [handle.cleanup.remote() for handle in handles],
                partial(ray.get, raise_on_error=False),
            )
        except Exception:
            pass
        try:
            await self._ray_get(self.registry.release_all_containers.remote())
        except Exception as e:
            return {"status": "failed", "error": repr(e)}
        return {"status": "ok", "container_ids": list(containers.keys())}

    def __del__(self):
        """Clean up this replica when it's destroyed"""
        # Signal background thread to stop, but don't wait
        self._stop_event.set()

        try:
            tasks = []
            tasks.append(self.registry.cleanup_replica.remote(self.replica_id))
            for handle in self.local_actors:
                try:
                    tasks.append(handle.cleanup.remote())
                except Exception:
                    pass
            ray.get(tasks, raise_on_error=False)
        except Exception:
            # Ignore all exceptions during cleanup
            pass


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments."""  # noqa: E501

    head_node = get_ray_head_node()

    # Get or create the detached global registry by name
    try:
        registry = ray.get_actor(ContainerRegistry.name, namespace=ACTOR_NAME_SPACE)
    except ValueError:
        registry = ContainerRegistry.options(  # type: ignore[attr-defined]
            name=ContainerRegistry.name,
            namespace=ACTOR_NAME_SPACE,
            lifetime="detached",
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=head_node["NodeID"],
                soft=False,
            ),
            num_cpus=0,
            num_gpus=0,
            max_restarts=3,  # Allow 3 automatic retries
            max_task_retries=-1,
        ).remote()

    return ContainerDeployment.options().bind(**cli_args)  # type: ignore[attr-defined]

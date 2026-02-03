import random
import time
from typing import List, Optional, Tuple

import fire
import ray

import matrix
from matrix.utils.ray import get_ray_address


def _init_ray():
    """Initialize Ray connection."""
    cli = matrix.Cli()
    cluster_info = cli.cluster.cluster_info()
    if cluster_info is None:
        raise RuntimeError("Cluster info not found. Is the cluster running?")
    address = get_ray_address(cluster_info)
    ray.init(
        address=address,
        log_to_driver=True,
        ignore_reinit_error=True,
    )


@ray.remote
def _kill_by_name_remote(
    actor_name: str,
    namespace: str,
    no_restart: bool = False,
) -> Tuple[bool, str]:
    """Remote function to kill an actor by name and namespace."""
    try:
        actor_handle = ray.get_actor(actor_name, namespace=namespace)
        ray.kill(actor_handle, no_restart=no_restart)
        return True, f"Killed actor (name={actor_name}, namespace={namespace})"
    except ValueError as e:
        return False, f"Actor not found (name={actor_name}, namespace={namespace}): {e}"
    except Exception as e:
        return (
            False,
            f"Failed to kill actor (name={actor_name}, namespace={namespace}): {e}",
        )


@ray.remote
def _kill_by_id_remote(actor_id: str, no_restart: bool = False) -> Tuple[bool, str]:
    """Remote function to kill an actor by ID."""
    from ray.experimental.state.api import list_actors

    actors = list_actors(filters=[("actor_id", "=", actor_id)])
    if not actors:
        return False, f"Actor not found (id={actor_id})"

    actor_info = actors[0]
    actor_name = actor_info.get("name")
    if not actor_name:
        return False, f"Actor has no name, cannot get handle (id={actor_id})"

    namespace = actor_info.get("ray_namespace")
    try:
        actor_handle = ray.get_actor(actor_name, namespace=namespace)
        ray.kill(actor_handle, no_restart=no_restart)
        return (
            True,
            f"Killed actor (id={actor_id}, name={actor_name}, namespace={namespace})",
        )
    except Exception as e:
        return False, f"Failed to kill actor (id={actor_id}): {e}"


def kill_actor_by_name(
    actor_name: str,
    namespace: str,
    no_restart: bool = False,
) -> bool:
    """
    Kill a Ray actor by name and namespace.

    Args:
        actor_name: Name of the actor to kill
        namespace: Ray namespace where the actor is located
        no_restart: If True, prevent the actor from restarting

    Returns:
        True if the actor was killed, False otherwise
    """
    _init_ray()

    success, msg = ray.get(
        _kill_by_name_remote.remote(actor_name, namespace, no_restart)  # type: ignore[call-arg]
    )
    if success:
        print(f"[OK] {msg}")
    else:
        print(f"[ERROR] {msg}")
    return success


def kill_actor_by_id(actor_id: str, no_restart: bool = False) -> bool:
    """
    Kill a Ray actor given its actor_id (hex string).

    Example: python kill_ray_actor.py by_id 3da829e2708207da97689d70af332a2ff5261092bd0dfcddda42434e

    Returns True if the actor was found and killed, False otherwise.
    """
    _init_ray()

    success, msg = ray.get(_kill_by_id_remote.remote(actor_id, no_restart))  # type: ignore[call-arg]
    if success:
        print(f"[OK] {msg}")
    else:
        print(f"[ERROR] {msg}")
    return success


def kill_random_repeatedly(
    actor_names: List[str],
    namespace: str,
    interval: float = 180.0,
    no_restart: bool = False,
    max_kills: Optional[int] = None,
) -> None:
    """
    Repeatedly kill a random actor from the list at specified intervals.

    Args:
        actor_names: List of actor names to randomly choose from
        namespace: Ray namespace where actors are located
        interval: Time in seconds between kills (default: 180 = 3 minutes)
        no_restart: If True, prevent actors from restarting
        max_kills: Maximum number of kills before stopping (None = infinite)

    Example:
        python kill_ray_actor.py random \
            --actor_names='["llm_agent_0", "llm_agent_1", "reward_agent_0"]' \
            --namespace="2024-01-15T10-30-00_0" \
            --interval=60
    """
    _init_ray()

    if not actor_names:
        print("[ERROR] No actor names provided")
        return

    print(f"Starting random actor killer:")
    print(f"  Actors: {actor_names}")
    print(f"  Namespace: {namespace}")
    print(f"  Interval: {interval}s")
    print(f"  No restart: {no_restart}")
    print(f"  Max kills: {max_kills or 'unlimited'}")
    print()

    kill_count = 0
    while max_kills is None or kill_count < max_kills:
        # Pick a random actor
        target = random.choice(actor_names)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Attempting to kill: {target}")

        # Kill remotely
        success, msg = ray.get(
            _kill_by_name_remote.remote(target, namespace, no_restart)  # type: ignore[call-arg]
        )
        if success:
            print(f"[OK] {msg}")
            kill_count += 1
        else:
            print(f"[FAILED] {msg}")

        if max_kills is None or kill_count < max_kills:
            print(f"Sleeping for {interval}s...")
            time.sleep(interval)

    print(f"\nDone. Total kills: {kill_count}")


if __name__ == "__main__":
    fire.Fire(
        {
            "by_name": kill_actor_by_name,
            "by_id": kill_actor_by_id,
            "random": kill_random_repeatedly,
        }
    )

import fire
import ray

import matrix
from matrix.utils.ray import get_ray_address


@ray.remote
def _kill_actor_remote(actor_id: str, no_restart: bool = False) -> tuple[bool, str]:
    """Remote function to kill an actor from within the Ray cluster."""
    from ray.experimental.state.api import list_actors

    actors = list_actors(filters=[("actor_id", "=", actor_id)])
    if not actors:
        return False, f"Actor not found (id={actor_id})"

    actor_info = actors[0]
    actor_name = actor_info.get("name")
    if not actor_name:
        return False, f"Actor has no name, cannot get handle (id={actor_id})"

    namespace = actor_info.get("ray_namespace")
    actor_handle = ray.get_actor(actor_name, namespace=namespace)
    ray.kill(actor_handle, no_restart=no_restart)
    return (
        True,
        f"Killed actor (id={actor_id}, name={actor_name}, namespace={namespace})",
    )


def kill_actor_by_id(actor_id: str, no_restart: bool = False) -> bool:
    """
    Kill a Ray actor given its actor_id (hex string).

    Example: python kill_ray_actor.py 3da829e2708207da97689d70af332a2ff5261092bd0dfcddda42434e

    Returns True if the actor was found and killed, False otherwise.
    """
    cli = matrix.Cli()
    address = get_ray_address(cli.cluster.cluster_info())
    ray.init(
        address=address,
        log_to_driver=True,
        ignore_reinit_error=True,
    )

    success, msg = ray.get(_kill_actor_remote.remote(actor_id, no_restart))
    if success:
        print(f"[OK] {msg}")
    else:
        print(f"[ERROR] {msg}")
    return success


if __name__ == "__main__":
    fire.Fire(kill_actor_by_id)

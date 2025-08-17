# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import fire
import ray


def delete_named_actor(ray_head_url: str, actor_name: str, namespace: str):
    assert ":" in ray_head_url, "ray_head_url should be in the format of hostname:port"
    if not ray_head_url.startswith("ray://"):
        ray_head_url = f"ray://{ray_head_url}"

    ray.init(
        address=ray_head_url,
    )

    try:
        actor = ray.get_actor(actor_name, namespace=namespace)
    except ValueError:
        print(f"No actor found with name {actor_name} and namespace {namespace}")
        return

    ray.kill(actor)
    print(f"Actor '{actor_name}' killed successfully.")


if __name__ == "__main__":
    fire.Fire(delete_named_actor)

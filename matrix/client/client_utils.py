# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import hashlib
import json
import random
import time
import typing as tp

from matrix.client.endpoint_cache import EndpointCache


async def get_an_endpoint_url(
    endpoint_cache: EndpointCache,
    multiplexed_model_id: str = "",
    force_update: bool = False,
) -> str:
    urls = await endpoint_cache(force_update)
    start_time = time.time()
    while not urls:
        # explicitly use synchronous sleep to block the whole event loop
        await asyncio.sleep(60)
        print(f"no worker is available, waited {time.time() - start_time}s..")
        urls = await endpoint_cache(force_update)

    if multiplexed_model_id:
        hashed_int = int(hashlib.sha256(multiplexed_model_id.encode()).hexdigest(), 16)
        return urls[hashed_int % len(urls)]
    else:
        return random.choice(urls)


def save_to_jsonl(
    data: tp.List[tp.Dict[str, tp.Any]],
    filename: str,
    write_mode: str,
    stats: tp.Dict[str, tp.Any],
) -> None:
    with open(filename, write_mode) as file:
        for item in data:
            stats["total"] += 1
            stats["success"] += 0 if "error" in item["response"] else 1
            stats["sum_latency"] += (
                item["response"]["response_timestamp"]
                - item["request"]["metadata"]["request_timestamp"]
            )
            json_str = json.dumps(item)
            file.write(json_str + "\n")

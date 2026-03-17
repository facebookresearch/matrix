# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import importlib
import json
import logging
import os
import re
import time
from collections import namedtuple
from dataclasses import MISSING, fields
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Type, TypeVar

import ray
from jinja2 import Template

if TYPE_CHECKING:
    from .orchestrator import Orchestrator
    from .sink import Sink


def setup_logging(logger, debug: bool):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    )
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    return logger


def get_ray_actor_class(target_path):
    """
    Get a Ray actor class from a target path, handling pre-decorated classes.
    """
    import ray

    module_path, class_name = target_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    actor_class = getattr(module, class_name)

    # Check if it's already a Ray actor
    if hasattr(actor_class, "remote"):
        return actor_class
    else:
        return ray.remote(actor_class)


def render_template(template: str, **kwargs) -> str:
    return Template(template).render(**kwargs, **os.environ)


T = TypeVar("T")


def extract_json(
    text: str,
    cls: Optional[Type[T]] = None,
) -> T | Dict[str, Any]:
    # Match fenced code block with optional json label
    match = re.search(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON block found in the text.")

    json_str = match.group(1)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    # If no dataclass, just return dict
    if cls is None:
        return data

    cls_fields = [f.name for f in fields(cls)]  # type: ignore[arg-type]
    mapped: Dict[str, Any] = {}
    unmapped_values: list[Any] = []

    # Step 1: direct + normalized name matching, collect unmapped values
    for k, v in data.items():
        norm_key = k.strip().lower().replace(" ", "_").replace("-", "_")
        found = False
        for fname in cls_fields:
            if fname == k or fname.lower() == norm_key:
                mapped[fname] = v
                found = True
                break
        if not found:
            unmapped_values.append(v)

    # Step 2: match remaining dataclass fields by order to unmapped values
    unmapped_fields = [f for f in cls_fields if f not in mapped]
    for fname, val in zip(unmapped_fields, unmapped_values):
        mapped[fname] = val

    # Step 3: fill missing required fields with None
    for f in fields(cls):  # type: ignore[arg-type]
        if (
            f.name not in mapped
            and f.default is MISSING
            and f.default_factory is MISSING
        ):
            mapped[f.name] = None

    return cls(**mapped)


class RayDict(dict):
    """
    dict subclass for auto Ray storage/dereference of specific large text fields.
    Optimized for fixed fields: "text", "output".
    """

    FIXED_FIELDS = [
        "text",
        "output",  # in history
    ]
    TEXT_SIZE_THRESHOLD = 512  # default size threshold

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str) and key in self.FIXED_FIELDS:
            return super().__contains__(key) or super().__contains__(f"{key}_ref")
        return super().__contains__(key)

    async def get_async(self, key: str, default: Any = None) -> Any:
        if key in self.FIXED_FIELDS:
            ref_key = f"{key}_ref"
            if ref_key in self:
                return await super().__getitem__(ref_key)
        return super().get(key, default)

    @classmethod
    async def from_dict(cls, data: Dict[str, Any], registry: "Sink") -> "RayDict":
        self = cls()
        for k, v in data.items():
            if (
                k in cls.FIXED_FIELDS
                and isinstance(v, str)
                and len(v) > cls.TEXT_SIZE_THRESHOLD
            ):
                # put with registry as owner otherwise ray will release when owner died even if there is reference.
                handle = ray.put(v, _owner=registry)
                self[f"{k}_ref"] = handle
                await registry.register_object.remote([handle])  # type: ignore[attr-defined]
            elif k in cls.FIXED_FIELDS:
                # store small text directly but bypass the assert
                super(cls, self).__setitem__(k, v)
            else:
                self[k] = v
        return self

    async def to_dict(self) -> dict[str, Any]:
        out = dict(self)
        for key in self.FIXED_FIELDS:
            ref_key = f"{key}_ref"
            if ref_key in out:
                out[key] = await out[ref_key]
                del out[ref_key]
        return out

    async def cleanup_ray(self, sink: "Sink"):
        refs_to_free = [
            self[f"{field}_ref"]
            for field in self.FIXED_FIELDS
            if f"{field}_ref" in self and self[f"{field}_ref"] is not None
        ]
        if refs_to_free:
            await sink.unregister_object(refs_to_free)  # type: ignore[attr-defined]
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: ray.internal.free(refs_to_free))


class HistPair(NamedTuple):
    agent: str
    response: RayDict


REMOTE_CALL_TIMEOUT = 60.0  # seconds per attempt
REMOTE_CALL_RETRIES = 3


async def send_with_retry(
    method,
    *args,
    timeout: float = REMOTE_CALL_TIMEOUT,
    max_retries: int = REMOTE_CALL_RETRIES,
    logger: Optional[logging.Logger] = None,
):
    """Call a Ray remote method with retry and timeout.

    Only retries on RayActorError (actor dead/restarting) and TimeoutError.
    Other exceptions are raised immediately.

    Args:
        method: Ray actor method (e.g. handle.submit)
        *args: Arguments to pass to method.remote(*args)
        timeout: Timeout per attempt in seconds
        max_retries: Maximum retry attempts
        logger: Logger for warnings

    Returns:
        The result of the remote call

    Raises:
        RuntimeError: If all retries exhausted
        Exception: Non-retryable exceptions are raised immediately
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return await asyncio.wait_for(method.remote(*args), timeout=timeout)
        except ray.exceptions.RayActorError as e:
            last_exception = e
            if logger:
                logger.warning(
                    f"Actor dead (attempt {attempt + 1}/{max_retries}): {repr(e)}"
                )
            continue
        except TimeoutError as e:
            last_exception = e
            if logger:
                logger.warning(
                    f"Timeout (attempt {attempt + 1}/{max_retries}): {repr(e)}"
                )
            continue
        except Exception:
            # Non-retryable, raise immediately
            raise

    raise RuntimeError(
        f"Remote call failed after {max_retries} attempts: {last_exception}"
    )

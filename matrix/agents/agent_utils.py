# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import json
import logging
import os
import re
from dataclasses import MISSING, fields
from typing import Any, Dict, Optional, Type, TypeVar

from jinja2 import Template


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

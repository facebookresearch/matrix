# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import re
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import matrix


def convert_to_json_compatible(obj: Any):
    if isinstance(obj, dict):
        return {
            str(key): convert_to_json_compatible(value) for key, value in obj.items()
        }
    if isinstance(obj, (list, tuple, set)):
        return [convert_to_json_compatible(item) for item in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        return convert_to_json_compatible(asdict(obj))
    if isinstance(obj, Enum):
        return convert_to_json_compatible(obj.value)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    return str(obj)


def sanitize_app_name(app_name: str) -> str:
    """Return a Ray Serve compatible application name.

    Ray Serve does not allow slashes in application names. This helper removes
    leading and trailing slashes and replaces any remaining ``/`` characters
    with ``-`` so that user provided names can be used safely.
    """

    return app_name.strip("/").replace("/", "-")


def get_user_message_from_llama3_prompt(text: str) -> str:
    PATTERN = re.compile(
        r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", re.DOTALL
    )

    if "<|end_header_id|>" in text:
        match = PATTERN.search(text)
        if not match:
            return text
        return match.group(1)
    else:
        return text


def str_to_callable(dotted_path: str):
    """
    Converts a dotted path string to a callable Python object.

    Args:
        dotted_path (str): Dotted path to the callable, e.g. 'matrix.job.job_utils.echo'

    Returns:
        callable: The resolved callable object

    Raises:
        ValueError: If the path is malformed, module can't be imported, or the object is not callable.
    """
    try:
        module_path, func_name = dotted_path.rsplit(".", 1)
    except ValueError:
        raise ValueError(
            f"Invalid path '{dotted_path}': Must be in 'module.attr' format."
        )

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Module '{module_path}' could not be imported. Original error:\n{e}"
        )

    try:
        obj = getattr(module, func_name)
    except AttributeError:
        raise ValueError(f"'{func_name}' not found in module '{module_path}'.")

    if not callable(obj):
        raise ValueError(f"'{dotted_path}' resolved to a non-callable object.")
    return obj


def get_nested_value(d, path: str, default: Any | None = None) -> Any:
    """Access nested ``dict``/``list`` using a dotted string path.

    When the path cannot be fully resolved, ``default`` is returned.
    Example::

        data = {"a": {"b": [1]}}
        get_nested_value(data, "a.b[0]")       # -> 1
        get_nested_value(data, "a.c", "miss")  # -> "miss"

    Args:
        d: Root dictionary/list to traverse.
        path: Dotted path string such as ``"a.b[0].c"``.
        default (Optional): Value returned when the path does not exist.

    Returns:
        The resolved value or ``default`` if any part of the path is missing.
    """

    tokens = re.findall(r"[^\[\].]+|\[\d+\]", path)
    try:
        for token in tokens:
            if token.startswith("[") and token.endswith("]"):
                index = int(token[1:-1])
                d = d[index]
            else:
                d = d[token]
        return d
    except (KeyError, IndexError, TypeError):
        return default

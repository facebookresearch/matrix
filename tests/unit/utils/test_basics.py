# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum

import pytest

from matrix.utils.basics import (
    convert_to_json_compatible,
    get_nested_value,
    get_user_message_from_llama3_prompt,
    sanitize_app_name,
    str_to_callable,
)


def test_sanitize_app_name():
    assert sanitize_app_name("meta-llama/Llama-3.1-8B") == "meta-llama-Llama-3.1-8B"
    assert sanitize_app_name("model") == "model"
    assert sanitize_app_name("a/b/c") == "a-b-c"
    assert sanitize_app_name("foo/bar/baz") == "foo-bar-baz"
    assert sanitize_app_name("/leading") == "leading"
    assert sanitize_app_name("trailing/") == "trailing"


class Color(Enum):
    RED = "red"
    BLUE = "blue"


@dataclass
class Demo:
    """Small dataclass used to validate JSON conversion."""

    a: int
    b: tuple[int, ...]
    c: set[str]
    color: Color


def test_convert_to_json_compatible_handles_types():
    data = {
        "int": 1,
        "float": 1.5,
        "bool": True,
        "none": None,
        "tuple": (1, 2),
        "set": {"x", "y"},
        "dc": Demo(5, (3, 4), {"z"}, Color.RED),
        "enum": Color.BLUE,
    }

    converted = convert_to_json_compatible(data)

    assert converted["int"] == 1
    assert converted["float"] == 1.5
    assert converted["bool"] is True
    assert converted["none"] is None
    assert converted["tuple"] == [1, 2]
    assert sorted(converted["set"]) == ["x", "y"]
    assert converted["dc"] == {"a": 5, "b": [3, 4], "c": ["z"], "color": "red"}
    assert converted["enum"] == "blue"


def test_convert_to_json_compatible_nested_structures():
    nested = {"items": [Demo(1, (2,), {"a"}, Color.RED)]}
    assert convert_to_json_compatible(nested) == {
        "items": [{"a": 1, "b": [2], "c": ["a"], "color": "red"}]
    }


def test_convert_to_json_compatible_dataclass_class():
    """Ensure dataclass *types* are stringified rather than treated as instances."""
    assert convert_to_json_compatible(Demo) == str(Demo)


def test_get_user_message_from_llama3_prompt():
    prompt = "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
    assert get_user_message_from_llama3_prompt(prompt) == "Hello"

    malformed = "<|start_header_id|>user<|end_header_id|>noeot"
    assert get_user_message_from_llama3_prompt(malformed) == malformed

    plain = "Just text"
    assert get_user_message_from_llama3_prompt(plain) == plain


def test_get_user_message_from_llama3_prompt_picks_first_user():
    conversation = (
        "<|start_header_id|>system<|end_header_id|>\n\nSys<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nHello<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nBye<|eot_id|>"
    )
    assert get_user_message_from_llama3_prompt(conversation) == "Hi"


def test_get_nested_value():
    data = {"a": {"b": [{"c": 3}, {"c": 4}]}, "x": [10, 20]}
    assert get_nested_value(data, "a.b[1].c") == 4
    assert get_nested_value(data, "x[0]") == 10
    assert get_nested_value(data, "a.b[2].c") is None
    assert get_nested_value(data, "a.b.c") is None
    assert get_nested_value(data, "missing.path", default="fallback") == "fallback"


def test_str_to_callable_success():
    func = str_to_callable("matrix.job.job_utils.echo")
    assert func("hi") == {"success": True, "output": "hi"}


@pytest.mark.parametrize(
    "path, message",
    [
        ("matrix.utils.ray.ACTOR_NAME_SPACE", "resolved to a non-callable object"),
        ("matrix.job.job_utils.missing", "not found in module"),
        ("matrix.missing.module", "could not be imported"),
        ("not.a.path", "could not be imported"),
        ("badpath", "Invalid path"),
    ],
)
def test_str_to_callable_errors(path, message):
    with pytest.raises(ValueError) as exc:
        str_to_callable(path)
    assert message in str(exc.value)

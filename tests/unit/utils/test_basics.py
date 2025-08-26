# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum

from matrix.utils.basics import convert_to_json_compatible, sanitize_app_name


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

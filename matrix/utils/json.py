# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
from dataclasses import asdict


def convert_to_json_compatible(obj):
    if isinstance(obj, dict):
        return {
            str(key): convert_to_json_compatible(value) for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_to_json_compatible(item) for item in obj]
    elif hasattr(obj, "__dataclass_fields__"):
        return convert_to_json_compatible(asdict(obj))
    else:
        return str(obj)


def get_user_prompt(text: str) -> str:
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

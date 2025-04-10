# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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

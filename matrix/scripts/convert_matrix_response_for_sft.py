# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import re

import fire
import pandas as pd

from matrix.utils.basics import get_nested_value


def main(
    input_jsonl: str,
    output_jsonl: str,
    src_key: str = "request.messages[0].content",
    target_key: str = "response.text[0]",
    src_template: str = "<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    target_template: str = "{text}",
):
    with open(input_jsonl) as inf:
        with open(output_jsonl, "w") as outf:
            for line in inf:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    src_text = get_nested_value(data, src_key)
                    tgt_text = get_nested_value(data, target_key)
                    src = src_template.format(text=src_text)
                    tgt = target_template.format(text=tgt_text)
                    outf.write(json.dumps({"src": src, "tgt": tgt}) + "\n")


if __name__ == "__main__":
    fire.Fire(main)

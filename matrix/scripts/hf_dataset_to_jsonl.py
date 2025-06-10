# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json

import fire
import tqdm
from datasets import load_dataset


def load_and_dump_dataset(
    dataset_name: str,
    split: str = "train",
    output_file: str = "dataset.jsonl",
    text_key: str | None = None,
    prompt_template: str | None = None,
):
    """
    Load dataset splits and dump them into a JSONL file.

    Args:
        dataset_name (str): Name of the dataset to load.
        split (str): dataset split to load.
        output_file (str): Path to the output JSONL file.
    """
    # Load dataset with specified splits
    dataset = load_dataset(dataset_name, split=split)

    # Write dataset to JSONL file
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in tqdm.tqdm(dataset):  # Iterate over each row
            if prompt_template and text_key:
                # If a prompt template is provided, format the text
                sample[text_key] = prompt_template.replace(
                    "<user_message>", sample[text_key]
                )
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Dataset saved to {output_file}")


if __name__ == "__main__":
    fire.Fire(load_and_dump_dataset)

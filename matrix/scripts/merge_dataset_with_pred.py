# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json

import fire

from matrix.utils.basics import get_nested_value


def load_and_dump_dataset(
    dataset_jsonl: str,
    pred_jsonl: str,
    output_file="dataset.jsonl",
    line_field: str = "request.metadata.line",
    pred_field: str = "response.text",
):
    """
    Load dataset splits and dump them into a JSONL file.

    Args:
        dataset_jsonl (str): Path to the dataset JSONL file.
        pred_jsonl (str): Path to the LLM response file.
        output_file (str): Path to the output JSONL file, taking everything from dataset and adding the response text as "pred".
        line_field (str): Dot-separated JSON field in pred_jsonl indicating the line number in dataset_jsonl (1-based index).
        pred_field (str): Dot-separated JSON field in pred_jsonl to copy to the output, renamed to "pred".
    """
    # Load dataset
    dataset = []
    with open(dataset_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(
                json.loads(line.strip())
            )  # Ensure we store each line as a JSON object

    # Load predictions
    predictions = {}
    with open(pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            pred_data = json.loads(line.strip())
            line_number = get_nested_value(pred_data, line_field)
            pred_text = get_nested_value(pred_data, pred_field)
            assert line_number is not None, f"Bad line {pred_data}"
            if pred_text is None:
                print(f"Failed line {pred_data}")
            else:
                predictions[int(line_number) - 1] = (
                    pred_text  # Convert to 0-based index
                )

    # Merge dataset with predictions
    missing = []
    with open(output_file, "w", encoding="utf-8") as f:
        for i, data in enumerate(dataset):
            if i in predictions:
                # only one pred
                data["pred"] = predictions[i]
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                missing.append(i)
    print(f"Dataset with predictions saved to {output_file}")
    print(f"Missing {len(missing)} lines {missing}")


if __name__ == "__main__":
    fire.Fire(load_and_dump_dataset)

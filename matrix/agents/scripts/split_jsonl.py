# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Split a JSONL file into multiple parts."""

import os
from pathlib import Path

import fire


def split_jsonl(input_file: str, num_split: int) -> None:
    """
    Split a JSONL file into multiple parts.

    Args:
        input_file: Path to the input JSONL file
        num_split: Number of parts to split into
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Read all lines
    with open(input_path, "r") as f:
        lines = f.readlines()

    total_lines = len(lines)
    if total_lines == 0:
        print("Input file is empty")
        return

    # Calculate lines per split
    base_size = total_lines // num_split
    remainder = total_lines % num_split

    # Generate output filenames
    stem = input_path.stem
    suffix = input_path.suffix
    parent = input_path.parent
    split_dir = parent / "split"
    split_dir.mkdir(parents=True, exist_ok=True)

    start = 0
    for i in range(num_split):
        # Distribute remainder across first splits
        size = base_size + (1 if i < remainder else 0)
        end = start + size

        output_path = split_dir / f"{stem}_part{i}{suffix}"
        with open(output_path, "w") as f:
            f.writelines(lines[start:end])

        print(f"Written {size} lines to {output_path}")
        start = end

    print(f"Split {total_lines} lines into {num_split} files")


if __name__ == "__main__":
    fire.Fire(split_jsonl)

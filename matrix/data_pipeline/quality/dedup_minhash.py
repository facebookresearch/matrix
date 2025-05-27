# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""minhash words to identify duplicates for the user prompt"""

import json
import os
import re
import shutil
import socket
import string
import time
import typing as tp
import uuid

import ray
import tqdm
from datasketch import LeanMinHash, MinHash, MinHashLSH
from fire import Fire

from matrix.utils.basics import get_user_message_from_llama3_prompt

num_perm = 128


def normalize_text(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def process_row(row, text_key):
    """Extract text and compute MinHash with index"""
    src_data = row[text_key]
    text = get_user_message_from_llama3_prompt(src_data)
    text = normalize_text(text)
    tokens = text.split()

    mh = MinHash(num_perm=num_perm)
    for token in tokens:
        mh.update(token.encode("utf-8"))

    lean_mh = LeanMinHash(mh)
    buf = bytearray(lean_mh.bytesize())
    lean_mh.serialize(buf)
    row["minhash"] = buf
    row["id"] = str(uuid.uuid4())

    return row


def minhash_from_values(mh_bytes: bytes):
    return LeanMinHash.deserialize(mh_bytes)


# don't run locally https://github.com/ray-project/ray/issues/35537
@ray.remote
def run_remotely(
    input_jsonl: str,
    output_dir: str,
    working_dir: str,
    max_concurrency: int,
    text_key: str,
    threshold=0.8,
):
    print(f"driver hostname is {socket.gethostname()}")
    start_time = time.time()

    ds = (
        ray.data.read_json(input_jsonl)  # type: ignore[attr-defined]
        .map(process_row, fn_kwargs={"text_key": text_key}, concurrency=max_concurrency)
        .filter(lambda row: row is not None)
    )
    ds.write_parquet(os.path.join(working_dir, "with_id"))

    # make sure id won't change
    ds = ray.data.read_parquet(os.path.join(working_dir, "with_id"))  # type: ignore[attr-defined]
    num_rows = ds.count()
    print(f"Total rows: {num_rows}")

    # Modified deduplication logic
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique_ids = set()
    sample_duplicates: tp.Dict[str, tp.List[str]] = {}

    def check_and_add(record):
        """Check if MinHash exists in LSH before adding"""
        id = record["id"]
        lean_minhash = minhash_from_values(record["minhash"])
        minhash = MinHash(seed=lean_minhash.seed, hashvalues=lean_minhash.hashvalues)

        duplicates = lsh.query(minhash)
        if not duplicates:  # No duplicates found
            lsh.insert(id, minhash)
            unique_ids.add(id)
        else:
            assert len(duplicates) >= 1, f"bad duplicates {duplicates}, current id {id}"
            duplicates = duplicates[0]

            if duplicates in sample_duplicates:
                sample_duplicates[duplicates].append(id)
            elif len(sample_duplicates) < 1000:
                sample_duplicates[duplicates] = [id]

    for record in tqdm.tqdm(
        ds.select_columns(["id", "minhash"]).iter_rows(), total=num_rows
    ):
        check_and_add(record)
    print(f"Unique rows: {len(unique_ids)}")

    (
        ds.filter(lambda row: row["id"] in unique_ids)
        .drop_columns(["id", "minhash"])
        .write_json(output_dir)
    )
    print(f"Time taken: {time.time() - start_time} seconds")

    # information about duplicates
    with open(
        os.path.join(working_dir, "sample_duplicates.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(sample_duplicates, f, ensure_ascii=False, indent=2)

    duplicate_ids = set(sample_duplicates.keys()).union(*sample_duplicates.values())
    (
        ds.filter(lambda row: row["id"] in duplicate_ids)
        .drop_columns(["minhash"])
        .write_json(os.path.join(working_dir, "sample_duplicates"))
    )


def main(
    ray_head_url: str,
    input_jsonl: str,
    output_dir: str,
    working_dir: str,
    max_concurrency: int = 40,
    text_key: str = "src",
    threshold=0.8,
):
    """Run minhash dedup on input jsonl.
    params:
    ray_head_url: Ray head (hostname:client_server_port), eg localhost:10001
    input_jsonl: file or dir of input jsonl.
    output_dir: name of the output directory.
    working_dir: name of the working directory for caching and debugging.
    max_concurrency: ray data concurrency.
    text_key: input json field for user prompt to dedup.
    threshold: dedup threshold, ie jaccard similarity
    """
    assert os.path.exists(input_jsonl), f"{input_jsonl} does not exist."
    assert not os.path.exists(output_dir), f"{output_dir} already exists."
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    assert ":" in ray_head_url, "ray_head_url should be in the format of hostname:port"
    if not ray_head_url.startswith("ray://"):
        ray_head_url = f"ray://{ray_head_url}"
    ray.init(address=ray_head_url, log_to_driver=True)
    ray.get(
        run_remotely.remote(
            input_jsonl,
            output_dir,
            working_dir,
            max_concurrency,
            text_key,
            threshold,
        )
    )


if __name__ == "__main__":
    Fire(main)

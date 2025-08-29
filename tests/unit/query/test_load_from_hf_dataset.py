# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest

pytest.importorskip("datasets")
from datasets import Dataset


def test_load_from_hf_dataset(monkeypatch):
    from matrix.client import query_llm

    dataset = Dataset.from_dict({"problem": ["1+1", "2+2"]})

    def mock_load_dataset(*args, **kwargs):
        return dataset

    monkeypatch.setattr("datasets.load_dataset", mock_load_dataset)

    lines = query_llm.load_from_hf_dataset(
        "dummy",
        "train",
        text_key="problem",
        messages_key="request.messages",
        system_prompt="sys",
    )

    assert len(lines) == 2
    assert lines[0]["messages"][0]["role"] == "system"
    assert lines[0]["messages"][0]["content"] == "sys"
    assert lines[0]["messages"][1]["content"] == "1+1"
    assert lines[0]["metadata"]["index"] == 0

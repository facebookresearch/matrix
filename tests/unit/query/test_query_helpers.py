# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import pytest

from matrix.client import query_llm


def test_convert_llama_instruct_text_basic():
    text = (
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n<|eot_id|>"
    )
    assert query_llm.convert_llama_instruct_text(text) == [
        {"role": "user", "content": "Hi"}
    ]


def test_convert_llama_instruct_text_keep_assistant():
    text = (
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n<|eot_id|>"
    )
    assert query_llm.convert_llama_instruct_text(text, keep_assistant=True) == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": ""},
    ]


def test_convert_llama_instruct_text_keep_assistant_content():
    text = (
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nAnswer<|eot_id|>"
    )
    assert query_llm.convert_llama_instruct_text(text, keep_assistant=True) == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Answer"},
    ]


def test_convert_llama_instruct_text_no_tokens():
    assert query_llm.convert_llama_instruct_text("Plain text") == [
        {"role": "user", "content": "Plain text"}
    ]


def test_convert_llama_instruct_text_error_on_trailing_assistant():
    text = (
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nAnswer<|eot_id|>"
    )
    with pytest.raises(AssertionError):
        query_llm.convert_llama_instruct_text(text)


def test_convert_llama_instruct_text_missing_eot():
    text = "<|start_header_id|>user<|end_header_id|>\n\nHi"
    assert query_llm.convert_llama_instruct_text(text) == [
        {"role": "user", "content": "Hi"}
    ]


def test_convert_llama_instruct_text_multi_turn():
    text = (
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nHello<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nBye<|eot_id|>"
    )
    assert query_llm.convert_llama_instruct_text(text) == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Bye"},
    ]


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def test_load_from_jsonl_with_prompt_and_metadata(tmp_path):
    file = tmp_path / "input.jsonl"
    _write_jsonl(
        file,
        [
            {"prompt": "Plain"},
            {
                "prompt": "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>",
                "metadata": {"id": 1},
            },
        ],
    )

    results = query_llm.load_from_jsonl(
        (str(file),), "prompt", "request.messages", system_prompt="sys"
    )

    assert len(results) == 2
    # First line uses default metadata
    assert results[0]["metadata"]["filename"] == str(file)
    assert results[0]["metadata"]["line"] == 1
    assert results[0]["messages"][0] == {"role": "system", "content": "sys"}
    # Second line keeps provided metadata
    assert results[1]["metadata"] == {"id": 1}
    assert results[1]["messages"][1]["content"] == "Hi"


def test_load_from_jsonl_with_messages_key(tmp_path):
    file = tmp_path / "input.jsonl"
    _write_jsonl(
        file,
        [
            {
                "request": {
                    "messages": [
                        {"role": "system", "content": "orig"},
                        {"role": "user", "content": "hi"},
                    ]
                }
            }
        ],
    )

    results = query_llm.load_from_jsonl(
        (str(file),), "prompt", "request.messages", system_prompt="new"
    )
    assert results[0]["messages"][0]["content"] == "new"
    assert results[0]["messages"][1]["content"] == "hi"


def test_load_from_jsonl_messages_no_metadata(tmp_path):
    file = tmp_path / "input.jsonl"
    _write_jsonl(
        file,
        [{"request": {"messages": [{"role": "user", "content": "hi"}]}}],
    )

    results = query_llm.load_from_jsonl(
        (str(file),), "prompt", "request.messages", system_prompt="sys"
    )

    assert results[0]["metadata"]["filename"] == str(file)
    assert results[0]["metadata"]["line"] == 1
    assert results[0]["messages"][0] == {"role": "system", "content": "sys"}
    assert results[0]["messages"][1] == {"role": "user", "content": "hi"}


def test_load_from_jsonl_invalid_json(tmp_path):
    file = tmp_path / "input.jsonl"
    file.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        query_llm.load_from_jsonl(
            (str(file),), "prompt", "request.messages", system_prompt=""
        )
    assert "Error in line 1" in str(exc.value)


def test_load_from_jsonl_missing_required_keys(tmp_path):
    file = tmp_path / "input.jsonl"
    _write_jsonl(file, [{"foo": "bar"}])

    with pytest.raises(ValueError) as exc:
        query_llm.load_from_jsonl(
            (str(file),), "prompt", "request.messages", system_prompt=""
        )
    assert "either prompt or request.messages should exist" in str(exc.value)


def test_convert_token_log_probs():
    class DummyVal:
        def __init__(self, logprob, rank, decoded_token):
            self.logprob = logprob
            self.rank = rank
            self.decoded_token = decoded_token

    class Dummy:
        def __init__(self, token_map):
            self.token_map = token_map

    token_map = {
        1: DummyVal(-0.5, 0, "hi"),
        2: DummyVal(-1.0, 1, "there"),
    }
    obj = Dummy(token_map)
    assert query_llm._convert_token_log_probs(obj) == {
        "1": {"logprob": -0.5, "rank": 0, "decoded_token": "hi"},
        "2": {"logprob": -1.0, "rank": 1, "decoded_token": "there"},
    }
    assert query_llm._convert_token_log_probs(Dummy({})) is None


def test_make_error_response():
    req = {"a": 1}
    resp = query_llm.make_error_response(req, ValueError("boom"))
    assert resp["request"] is req
    assert "boom" in resp["response"]["error"]
    assert "response_timestamp" in resp["response"]

    resp2 = query_llm.make_error_response(req, None)
    assert resp2["response"]["error"] == "unknown error"


def test_get_request_and_metadata_key():
    data = {"a": {"b": {"c": 1}}}
    assert query_llm._get_request("a.b.c", data) == 1
    assert query_llm._get_request("a.x.c", data) is None
    assert query_llm._get_metadata_key("a.b.c") == "a.b.metadata"

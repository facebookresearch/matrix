# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from google.protobuf import json_format

from matrix.app_server.llm import openai_pb2


def _build_response_dict():
    return {
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "text": "hello",
                "logprobs": None,
                "finish_reason": "stop",
                "prompt_logprobs": [{1: {"logprob": -0.1}}],
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    }


def test_prompt_logprob_token_ids_are_strings():
    response_dict = _build_response_dict()

    # Parsing fails when integer keys are used in protobuf map
    with pytest.raises(json_format.ParseError):
        json_format.ParseDict(response_dict, openai_pb2.CompletionResponse())

    # Apply the same transformation as GrpcDeployment.CreateCompletion
    choice = response_dict["choices"][0]
    prompt_logprobs = choice.get("prompt_logprobs")
    assert prompt_logprobs is not None
    for index, logprobs in enumerate(prompt_logprobs):
        token_map = {str(token_id): info for token_id, info in (logprobs or {}).items()}
        prompt_logprobs[index] = {"token_map": token_map}

    proto = openai_pb2.CompletionResponse()
    json_format.ParseDict(response_dict, proto)
    assert proto.choices[0].prompt_logprobs[0].token_map[1].logprob == pytest.approx(
        -0.1
    )


def test_prompt_logprobs_missing_parses():
    response_dict = _build_response_dict()
    response_dict["choices"][0].pop("prompt_logprobs")

    proto = openai_pb2.CompletionResponse()
    json_format.ParseDict(response_dict, proto)
    assert proto.choices[0].prompt_logprobs == []

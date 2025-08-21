# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from unittest.mock import MagicMock, patch

import pytest

import matrix
from matrix.client import query_llm


def test_batch_requests_from_async_run():
    """Test batch_requests called from within an asyncio.run context."""
    mock_response = "mocked_response"

    async def mock_make_request_async(_url, _model, data):
        return {"response": {"text": f"{mock_response}_{data}"}}

    async def async_wrapper():
        with patch(
            "matrix.client.query_llm.make_request",
            side_effect=mock_make_request_async,
        ):
            requests = [1, 2, 3]
            # batch_requests should handle the async context internally
            # and return a list directly, not a task
            result = query_llm.batch_requests("", "", requests)

            # Verify it returned a list, not a task
            assert isinstance(result, list)
            assert len(result) == 3
            assert result == [
                {"response": {"text": text}}
                for text in [
                    f"{mock_response}_1",
                    f"{mock_response}_2",
                    f"{mock_response}_3",
                ]
            ]

    # Use asyncio.run to execute the async wrapper
    asyncio.run(async_wrapper())


def test_batch_requests_in_sync_context():
    """Test batch_requests when called from a synchronous context."""
    # Create a mock for make_request_async
    mock_response = "mocked_response"

    async def mock_make_request_async(_url, _model, data):
        return {"response": {"text": f"{mock_response}_{data}"}}

    with patch(
        "matrix.client.query_llm.make_request",
        side_effect=mock_make_request_async,
    ):
        # Test with a list of requests
        requests = [1, 2, 3]
        result = query_llm.batch_requests("", "", requests)

        # Verify results
        assert len(result) == 3
        assert result == [
            {"response": {"text": text}}
            for text in [
                f"{mock_response}_1",
                f"{mock_response}_2",
                f"{mock_response}_3",
            ]
        ]


def test_batch_requests_empty_list():
    """Test batch_requests with an empty list."""
    with patch("matrix.client.query_llm.make_request") as mock_request:
        result = query_llm.batch_requests("", "", [])
        # make_request_async should not be called
        mock_request.assert_not_called()
        # Result should be an empty list
        assert result == []

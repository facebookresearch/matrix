# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from unittest.mock import MagicMock, patch

import pytest

import matrix
from matrix.app_server.llm import query_llm


@pytest.mark.asyncio
async def test_batch_requests_in_async_context():
    """Test batch_requests when called from an async context."""
    # Create a mock for make_request_async
    mock_response = "mocked_response"

    async def mock_make_request_async(_url, _model, request):
        return f"{mock_response}_{request}"

    with patch(
        "matrix.app_server.llm.query_llm.make_request",
        side_effect=mock_make_request_async,
    ):
        # Test with a list of requests
        requests = [1, 2, 3]
        result = await query_llm.batch_requests("", "", requests)

        # Verify results
        assert len(result) == 3
        assert result == [
            f"{mock_response}_1",
            f"{mock_response}_2",
            f"{mock_response}_3",
        ]


def test_batch_requests_in_sync_context():
    """Test batch_requests when called from a synchronous context."""
    # Create a mock for make_request_async
    mock_response = "mocked_response"

    async def mock_make_request_async(_url, _model, request):
        return f"{mock_response}_{request}"

    with patch(
        "matrix.app_server.llm.query_llm.make_request",
        side_effect=mock_make_request_async,
    ):
        # Test with a list of requests
        requests = [1, 2, 3]
        result = query_llm.batch_requests("", "", requests)

        # Verify results
        assert len(result) == 3
        assert result == [
            f"{mock_response}_1",
            f"{mock_response}_2",
            f"{mock_response}_3",
        ]


@pytest.mark.asyncio
async def test_batch_requests_empty_list():
    """Test batch_requests with an empty list."""
    with patch("matrix.app_server.llm.query_llm.make_request") as mock_request:
        result = await query_llm.batch_requests("", "", [])
        # make_request_async should not be called
        mock_request.assert_not_called()
        # Result should be an empty list
        assert result == []

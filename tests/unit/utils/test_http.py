# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web

from matrix.utils.http import fetch_url, fetch_url_sync, post_url


@pytest.mark.asyncio
async def test_fetch_url(unused_tcp_port):
    """Verify async HTTP helper for GET requests."""

    async def handle_get(request):
        return web.Response(text="hello", status=200)

    app = web.Application()
    app.router.add_get("/", handle_get)

    runner = web.AppRunner(app)
    await runner.setup()
    port = unused_tcp_port
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    try:
        url = f"http://127.0.0.1:{port}/"
        status, content = await fetch_url(url)
        assert status == 200
        assert content == "hello"
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_fetch_url_handles_errors(unused_tcp_port):
    """Ensure fetch_url gracefully handles network errors."""

    status, content = await fetch_url(f"http://127.0.0.1:{unused_tcp_port}")
    assert status is None
    assert "Unexpected error" in content


@pytest.mark.asyncio
async def test_post_url(unused_tcp_port):
    """Verify async POST helper sends data correctly."""

    async def handle_post(request):
        payload = await request.json()
        return web.json_response(payload)

    app = web.Application()
    app.router.add_post("/", handle_post)

    runner = web.AppRunner(app)
    await runner.setup()
    port = unused_tcp_port
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    try:
        async with aiohttp.ClientSession() as session:
            status, content = await post_url(
                session, f"http://127.0.0.1:{port}/", {"foo": "bar"}
            )
        assert status == 200
        assert json.loads(content) == {"foo": "bar"}
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_post_url_handles_errors():
    """Ensure post_url surfaces unexpected errors."""

    async with aiohttp.ClientSession() as session:
        with patch.object(session, "post", side_effect=Exception("boom")):
            status, content = await post_url(session, "http://example.com")

    assert status is None
    assert "boom" in content


def test_fetch_url_sync(unused_tcp_port):
    """Ensure synchronous fetch works and handles errors."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format, *args):  # pragma: no cover
            pass

    port = unused_tcp_port
    server = HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        status, content = fetch_url_sync(f"http://127.0.0.1:{port}")
        assert status == 200
        assert content == "ok"
    finally:
        server.shutdown()
        thread.join()

    with patch("requests.get") as mock_get:
        mock_get.side_effect = Exception("boom")
        status, content = fetch_url_sync("http://127.0.0.1:1")
        assert status is None
        assert "boom" in content

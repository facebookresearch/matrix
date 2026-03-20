# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import aiohttp
import requests


DEFAULT_HTTP_TIMEOUT_SECONDS = 30


async def post_url(session, url, data=None, timeout=DEFAULT_HTTP_TIMEOUT_SECONDS):
    """Send a POST request and return ``(status_code, text_content)``.

    A default timeout is applied to avoid indefinitely hanging control-plane calls.
    """
    try:
        async with session.post(url, json=data, timeout=timeout) as response:
            status = response.status  # Get the HTTP status code
            content = await response.text()  # Get the response body as text
            return status, content
    except Exception as e:
        return None, repr(e)  # Return None for status and error message as content


async def fetch_url(url, headers=None, timeout=DEFAULT_HTTP_TIMEOUT_SECONDS):
    """Asynchronously fetch a URL and return ``(status_code, text_content)``.

    A default timeout is applied to avoid indefinitely hanging health checks.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=timeout) as response:
                status = response.status  # Get the status code
                content = await response.text()  # Get response body as text
                return status, content
    except Exception as e:  # Catch-all for unexpected errors
        return None, f"Unexpected error: {str(e)}"


def fetch_url_sync(url, headers=None, timeout=DEFAULT_HTTP_TIMEOUT_SECONDS):
    """Synchronously fetch a URL and return ``(status_code, text_content)``.

    A default timeout is applied to avoid indefinitely hanging health checks.
    """
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        status = response.status_code  # Get the status code
        content = response.text  # Get response body as text
        return status, content
    except Exception as e:  # Catch-all for unexpected errors
        return None, f"Unexpected error: {str(e)}"

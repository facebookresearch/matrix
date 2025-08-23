# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import aiohttp
import requests


async def post_url(session, url, data=None):
    """Send a POST request to a given URL with optional data, returning status and content."""
    try:
        async with session.post(url, json=data) as response:
            status = response.status  # Get the HTTP status code
            content = await response.text()  # Get the response body as text
            return status, content
    except Exception as e:
        return None, repr(e)  # Return None for status and error message as content


async def fetch_url(url, headers=None):
    """Asynchronously fetch data from a single URL, returning status code and content."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                status = response.status  # Get the status code
                content = await response.text()  # Get response body as text
                return status, content
    except Exception as e:  # Catch-all for unexpected errors
        return None, f"Unexpected error: {str(e)}"


def fetch_url_sync(url, headers=None):
    """Synchronously fetch data from a single URL, returning status code and content."""
    try:
        response = requests.get(url, headers=headers)
        status = response.status_code  # Get the status code
        content = response.text  # Get response body as text
        return status, content
    except Exception as e:  # Catch-all for unexpected errors
        return None, f"Unexpected error: {str(e)}"

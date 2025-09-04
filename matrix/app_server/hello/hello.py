# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import ray
import requests
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse


# 1: Define a Ray Serve application.
@serve.deployment
class HelloDeployment:
    async def __call__(self, request: Request) -> JSONResponse:
        return JSONResponse({"result": "Hello"}, status_code=200)


app = HelloDeployment.bind()  # type: ignore[attr-defined]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import typing as tp

import requests
import starlette.requests
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

from matrix.app_server.code.sandbox_runner import SandboxRunner

CODE_EXEC_TIMEOUT = 10
logger = logging.getLogger("ray.serve")


@serve.deployment(ray_actor_options={"num_cpus": 1, "num_gpus": 0})
class CodeExecutionApp:
    async def __call__(self, request: Request) -> JSONResponse:

        if request.method == "POST":
            request_json = await request.json()  # Access the POST data
            code = request_json.get("code")
            timeout = request_json.get("timeout", CODE_EXEC_TIMEOUT)

        else:
            return JSONResponse(
                {"error": "Data must be in JSON format!"}, status_code=400
            )

        # run the code in sandbox
        try:
            runner = SandboxRunner(
                shared_paths=[],
                resource_limits={"time": timeout},
            )
            result = runner.run(code)
            if result["success"]:
                return JSONResponse(result, status_code=200)
            else:
                return JSONResponse(result, status_code=200)
        except Exception as e:
            return JSONResponse(
                {"success": False, "error_msg": str(e)}, status_code=500
            )


app = CodeExecutionApp.bind()  # type: ignore[attr-defined]

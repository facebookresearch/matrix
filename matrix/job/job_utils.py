# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import uuid

from matrix.utils.ray import Action


def generate_job_id():
    """Generates a unique job ID."""
    return f"job_{uuid.uuid4()}"


def generate_task_id(job_id, index):
    """Generates a unique task ID within a job."""
    return f"{job_id}_task_{index}"


class JobNotFound(Exception):
    """Raised when a job ID is not found in the manager."""

    pass


class JobAlreadyExist(Exception):
    """Raised when a job ID already exist."""

    pass


class RayJobManagerError(Exception):
    """Base exception for the package."""

    pass


class ActorUnavailableError(RayJobManagerError):
    """Raised when the JobManager actor cannot be reached."""

    pass


def echo(text):
    return {"success": True, "output": text}


def deploy_helper(app_api, applications):
    """helper functions to do deployment for jobs"""
    if applications:
        return app_api.deploy(Action.ADD, applications)
    return []


def check_status_helper(app_api, app):
    return app_api.app_status(app["name"])


def undeploy_helper(app_api, app):
    return app_api.deploy(Action.REMOVE, app)

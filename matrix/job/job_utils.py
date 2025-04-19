import json
import uuid

from matrix.utils.ray import Action


def generate_job_id():
    """Generates a unique job ID."""
    return f"job_{uuid.uuid4()}"


def generate_task_id(job_id, index):
    """Generates a unique task ID within a job."""
    return f"{job_id}_task_{index}"


def serialize_func(func):
    """Placeholder for potential future complex serialization if needed.
    For now, Ray handles function serialization directly.
    """
    # Basic check: ensure it's callable
    if not callable(func):
        raise TypeError("Provided 'func' must be callable.")
    return func  # Ray handles the actual serialization


def deserialize_func(serialized_func):
    """Placeholder for potential future complex deserialization."""
    return serialized_func  # Ray handles this


def is_json_serializable(data):
    """Checks if data is JSON serializable."""
    if data is None:
        return True
    try:
        json.dumps(data)
        return True
    except (TypeError, OverflowError):
        return False


# ray_task_manager/exceptions.py
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
    return True, text


def status_is_success(app_status: str) -> bool:
    return app_status == "RUNNING"


def status_is_failure(app_status: str) -> bool:
    return app_status in ["DEPLOY_FAILED", "DELETING"]


def status_is_pending(app_status: str) -> bool:
    return app_status in ["NOT_STARTED", "DEPLOYING", "UNHEALTHY"]


def deploy_helper(app_api, applications):
    """helper functions to do deployment for jobs"""
    if applications:
        return app_api.deploy(Action.ADD, applications)
    return []


def check_status_helper(app_api, app):
    return app_api.app_status(app["name"])


def undeploy_helper(app_api, app):
    return app_api.deploy(Action.REMOVE, app)

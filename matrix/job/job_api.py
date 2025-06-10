# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import partial

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

import matrix
from matrix.app_server.deploy_utils import validate_applications
from matrix.common import JOB_MANAGER_STORE
from matrix.job.job_manager import ACTOR_NAME, NAMESPACE, JobManager
from matrix.job.job_utils import (
    ActorUnavailableError,
    JobNotFound,
    RayJobManagerError,
    check_status_helper,
    deploy_helper,
    generate_job_id,
    generate_task_id,
    undeploy_helper,
)
from matrix.utils.basics import str_to_callable
from matrix.utils.ray import Action, get_ray_address, get_ray_head_node

logger = logging.getLogger(__name__)


class JobApi:

    def __init__(self, cluster_dir, cluster_info, app):
        self._cluster_dir = cluster_dir
        self._cluster_info = cluster_info
        self._app = app
        self._job_manager = None

    def _get_manager_actor(self, get_only=False):  # Add optional concurrency arg
        """Gets a handle to the JobManager actor, creating it if necessary."""
        if self._job_manager is not None:
            return self._job_manager

        if not ray.is_initialized():
            ray.init(
                address=get_ray_address(self._cluster_info),
                ignore_reinit_error=True,
                log_to_driver=False,
            )
        try:
            # Try getting existing actor first
            self._job_manager = ray.get_actor(ACTOR_NAME, namespace=NAMESPACE)
            logger.debug(f"Retrieved existing JobManagerActor handle.")
            # _actor_handle = actor # Cache if using global
            return self._job_manager
        except ValueError:
            logger.info(
                f"JobManagerActor '{ACTOR_NAME}' not found in namespace '{NAMESPACE}', attempting to create..."
            )
            if get_only:
                raise
            try:
                # Create the actor with the specified (or default) concurrency
                # Pass the max_concurrency value during creation if provided
                head_node = get_ray_head_node()

                actor_options = {
                    "name": ACTOR_NAME,
                    "namespace": NAMESPACE,
                    "lifetime": "detached",
                    "scheduling_strategy": NodeAffinitySchedulingStrategy(
                        node_id=head_node["NodeID"],
                        soft=False,
                    ),
                }

                # Create remotely with constructor args
                self._job_manager = JobManager.options(**actor_options).remote(  # type: ignore[attr-defined]
                    checkpoint_path=(self._cluster_dir / JOB_MANAGER_STORE)
                )

                # Ping the actor to ensure it started okay
                ray.get(self._job_manager.get_all_job_ids.remote())
                logger.info(f"Created new JobManagerActor.")
                # _actor_handle = actor # Cache if using global
                return self._job_manager
            except Exception as create_exc:
                logger.error(
                    f"Failed to create JobManager actor: {create_exc}", exc_info=True
                )
                raise ActorUnavailableError(
                    f"Could not create JobManager actor: {create_exc}"
                ) from create_exc
        except Exception as e:
            logger.error(f"Failed to get JobManager actor handle: {e}", exc_info=True)
            raise ActorUnavailableError(
                f"Could not get JobManager actor handle: {e}"
            ) from e

    def submit(
        self,
        job_definition,
    ):
        """
        Submits tasks. Optionally sets max concurrency if the actor is created here.

        job_definitions: dict like:
          {
            'job_id": str
            'max_concurrent_tasks': int
            'timeout': int
            'applications': List[matrix applications] to deploy,
            'deploy_applications': callable (optional),
            'check_status': callable (optional),
            'cleanup_applications': callable (optional),
          }
        task_definitions: list of dicts like:
          {
            'task_id': str,
            'func': callable,
            'args': tuple,
            'kwargs': dict,
            'resources': dict (e.g., {'CPU': 1, 'GPU': 0}),
            'applications': List[matrix applications] to deploy,
          }

        """

        assert "task_definitions" in job_definition

        default_params = {
            "job_id": generate_job_id(),
            "max_concurrent_tasks": 1,
            "timeout": 3600,
            "applications": [],
            "deploy_applications": partial(deploy_helper, self._app),
            "check_status": partial(check_status_helper, self._app),
            "cleanup_applications": partial(undeploy_helper, self._app),
        }
        for k, v in default_params.items():
            if k not in job_definition:
                job_definition[k] = v

        task_definitions = job_definition["task_definitions"]
        if not isinstance(task_definitions, list):
            raise TypeError("task_definitions must be a list")
        validate_applications(job_definition["applications"])
        for i, task_def in enumerate(task_definitions):
            if not isinstance(task_def, dict):
                raise TypeError(f"Item {i} not dict")
            if "func" not in task_def:
                raise ValueError(f"Task {i} missing 'func'")
            if isinstance(task_def["func"], str):
                task_def["func"] = str_to_callable(task_def["func"])
            if not callable(task_def["func"]):
                func = task_def["func"]
                raise TypeError(
                    f"Provided 'func' {str(func)} in {task_def} must be callable."
                )

            default_params = {
                "resources": {"CPU": 1},
                "applications": [],
                "args": (),
                "kwargs": {},
                "task_id": generate_task_id(job_definition["job_id"], i),
            }
            for k, v in default_params.items():
                if k not in task_def:
                    task_def[k] = v
            validate_applications(task_def["applications"])

        job_id = job_definition["job_id"]
        logger.info(f"Submitting job {job_id} via API.")

        actor = self._get_manager_actor()

        try:
            submitted_job_id = ray.get(actor.submit_job.remote(job_definition))
            logger.info(f"Job {submitted_job_id} successfully submitted to actor.")
            return submitted_job_id
        except Exception as e:
            logger.error(f"Failed to submit job {job_id} to actor: {e}", exc_info=True)
            raise RayJobManagerError(
                f"Actor communication failed during submission: {e}"
            ) from e

    def status(self, job_id):
        """
        Retrieves the current status of a previously submitted job.

        Args:
            job_id (str): The ID of the job to check.
            ray_address (str): Ray cluster address ('auto' to connect or start).
            **ray_init_kwargs: Additional keyword arguments for ray.init().

        Returns:
            dict: A dictionary containing the job status and progress details.
                Example: {'job_id': 'job_xyz', 'status': 'RUNNING', 'total_tasks': 10, ...}

        Raises:
            JobNotFound: If the job ID does not exist.
            RayJobManagerError: If Ray initialization or actor communication fails.
            ActorUnavailableError: If the manager actor cannot be reached.
        """
        actor = self._get_manager_actor()  # Get existing actor
        try:
            status = ray.get(actor.get_job_status.remote(job_id))
            return status
        # ... (rest of error handling as before) ...
        except JobNotFound:
            raise
        except Exception as e:
            logger.error(f"Failed to get status for job {job_id}: {e}", exc_info=True)
            if isinstance(e, ray.exceptions.RayActorError):
                raise Exception(f"Actor error getting status {job_id}: {e}") from e
            raise Exception(f"Failed to get job status: {e}") from e

    def get_results(self, job_id):
        """
        Retrieves the results for all tasks within a job.

        Should typically be called after the job status is COMPLETED or FAILED.
        If called while RUNNING, results for completed tasks will be returned,
        and placeholders or errors for tasks not yet finished.

        Args:
            job_id (str): The ID of the job to retrieve results for.
            ray_address (str): Ray cluster address ('auto' to connect or start).
            **ray_init_kwargs: Additional keyword arguments for ray.init().

        Returns:
            dict: A dictionary mapping task_id to its result tuple (bool, json_data).
                Example: {'job_xyz_task_0': {'success': True, 'output': 123}),
                            'job_xyz_task_1': {'success': False, 'error': 'Something failed'}}

        Raises:
            JobNotFound: If the job ID does not exist.
            RayJobManagerError: If Ray initialization or actor communication fails.
            ActorUnavailableError: If the manager actor cannot be reached.
        """
        actor = self._get_manager_actor()

        try:
            results = ray.get(actor.get_job_results.remote(job_id))
            return results
        except JobNotFound:  # Catch specific exception from actor
            raise
        except Exception as e:
            logger.error(
                f"Failed to get results for job {job_id} from actor: {e}", exc_info=True
            )
            if isinstance(e, ray.exceptions.RayActorError):
                raise RayJobManagerError(
                    f"Actor error while getting results for job {job_id}: {e}"
                ) from e
            raise RayJobManagerError(f"Failed to get job results: {e}") from e

    def list(self):
        """Lists all job IDs known by the JobManager."""
        actor = self._get_manager_actor()
        try:
            job_ids = ray.get(actor.get_all_job_ids.remote())
            return job_ids
        except Exception as e:
            logger.error(f"Failed to list jobs from actor: {e}", exc_info=True)
            raise RayJobManagerError(f"Failed to list jobs: {e}") from e

    def delete(self, job_id):
        """Remove the given `job_id`."""
        actor = self._get_manager_actor()
        try:
            ray.get(actor.delete_job.remote(job_id))
        except Exception as e:
            logger.error(f"Failed to delete job from actor: {e}", exc_info=True)
            raise RayJobManagerError(f"Failed to delete jobs: {e}") from e

    def clear(self):
        """Remove all jobs."""
        job_ids = self.list()
        for job in job_ids:
            self.delete(job)
        return job_ids

    def start(self):
        """Start the JobManager actor."""
        self._get_manager_actor()

    def stop(self):
        """Attempts to gracefully shut down the JobManager actor."""
        logger.info(
            f"Attempting to shutdown JobManager actor ({ACTOR_NAME} in {NAMESPACE})..."
        )
        try:
            actor = self._get_manager_actor(get_only=True)
            ray.get(actor.stop.remote())
            ray.kill(actor)
            logger.info("JobManager actor killed.")
        except ValueError:
            logger.info("JobManager actor not found, likely already shut down.")
        except Exception as e:
            logger.error(f"Error during JobManager shutdown: {e}", exc_info=True)
            raise RayJobManagerError(f"Failed to shutdown manager: {e}") from e

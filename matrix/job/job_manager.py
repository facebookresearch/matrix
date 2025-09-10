# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import inspect
import logging
import math
import os
import pickle
import socket
import sys
import threading
import time
import traceback
import typing as tp
from collections import defaultdict
from enum import Enum
from types import SimpleNamespace
from typing import cast

import ray

from matrix.job.job_utils import JobAlreadyExist, JobNotFound, generate_task_id
from matrix.utils.ray import status_is_failure, status_is_pending, status_is_success

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
MAX_RETRIES = 3
ACTOR_NAME = "system.JobManagerActor"
NAMESPACE = "matroid"
STATUS_CHECK_INTERVAL = 10  # run app status check every 10 seconds
JOB_MONITOR_INTERVAL = 5  # run the monitor loop evey 5 seconds


# --- Enums for Status ---
class TaskStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"


class JobStatus(Enum):
    SUBMITTED = "SUBMITTED"
    DEPLOYING = "DEPLOYING"
    STATUSCHECKING = "STATUSCHECKING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# --- Ray Remote Function (Worker Task) ---
# max_retries is 0 because the actor handles retries manually
@ray.remote(max_retries=0)
def _execute_task_sequence(
    user_func,
    args,
    kwargs,
    applications,
    deploy_func,
    check_status_func,
    cleanup_func,
    timeout_s,
    task_id,  # Pass task_id for logging on the worker
) -> tp.Dict[str, tp.Any]:
    """
    Wrapper to execute the deploy, health check, user function, and cleanup sequence.
    This runs as a single Ray task on a worker.
    """
    logger_actor = ray.get_actor(ACTOR_NAME, NAMESPACE)
    log_fn = logger_actor.log.remote
    log_fn(f"[{task_id}] Starting task execution sequence on {socket.gethostname()}")

    try:
        # --- Step 1: Deploy Applications ---
        log_fn(f"[{task_id}] Starting deployment...")
        try:
            deployment_result = deploy_func(applications)
            log_fn(f"[{task_id}] Deployment finished. Result: {deployment_result}")
        except Exception as e:
            error_tb = traceback.format_exc()
            log_fn(f"[{task_id}] Deployment failed: {e}\n{error_tb}")
            return {
                "success": False,
                "step": "deploy",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": error_tb,
            }

        # --- Step 2: Health Check ---
        if check_status_func:

            def report_failure(failed_apps):
                log_fn(
                    f"[{task_id}] Health check failed for apps {failed_apps} after {max_status_check_attempts} attempts."
                )
                return {
                    "success": False,
                    "step": "status_check",
                    "error_type": "HealthCheckFailure",
                    "error_message": f"Applications at indices {failed_apps} did not become healthy within limits.",
                }

            log_fn(f"[{task_id}] Starting health check...")
            max_status_check_attempts = math.ceil(timeout_s / STATUS_CHECK_INTERVAL)

            # Track which applications have succeeded
            app_success = [False] * len(applications)
            for attempt in range(1, max_status_check_attempts + 1):
                # Check each application that hasn't succeeded yet
                for i, app in enumerate(applications):
                    if app_success[i]:  # Skip successful apps
                        continue

                    try:
                        status = check_status_func(app)
                        log_fn(
                            f"[{task_id}] Health check attempt {attempt}: {app} {status}"
                        )

                        if status_is_success(status):
                            app_success[i] = True
                        elif status_is_failure(status):
                            return report_failure([app])
                    except Exception as e:
                        log_fn(
                            f"[{task_id}] Health check failed for app {i} on attempt {attempt}: {e}"
                        )

                # Check if all apps succeeded
                if all(app_success):
                    break

                time.sleep(STATUS_CHECK_INTERVAL)  # Wait before next attempt

            # After all attempts, check for failed apps
            failed_apps = [
                applications[i] for i, success in enumerate(app_success) if not success
            ]
            if failed_apps:
                return report_failure(failed_apps)

        # --- Step 3: Execute User Function ---
        log_fn(f"[{task_id}] Starting user function execution...")
        try:
            if isinstance(user_func, functools.partial):
                sig = inspect.signature(user_func.func)
            else:
                sig = inspect.signature(user_func)
            if "logging_config" in sig.parameters:
                kwargs = kwargs.copy()
                kwargs["logging_config"] = {"remote": True, "logger": logger_actor}

            user_result = user_func(*args, **kwargs)

            if not isinstance(user_result, dict) or "success" not in user_result:
                raise TypeError(
                    f"User function {user_func} did not return a dic containing success flag."
                )

            log_fn(f"[{task_id}] User function finished. {user_result}")
            return user_result

        except Exception as e:
            error_tb = traceback.format_exc()
            log_fn(f"[{task_id}] Exception in user task execution: {e}\n{error_tb}")
            return {
                "success": False,
                "step": "user_function",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": error_tb,
            }

    finally:
        # --- Step 4: Cleanup Applications ---
        if cleanup_func:
            log_fn(f"[{task_id}] Starting cleanup...")
            try:
                cleanup_results = cleanup_func(applications)
                log_fn(
                    f"[{task_id}] Cleanup finished successfully removed {cleanup_results}."
                )
            except Exception as e:
                error_tb = traceback.format_exc()
                log_fn(f"[{task_id}] Cleanup failed: {e}\n{error_tb}")


# --- Job Manager Actor ---
@ray.remote(num_cpus=0)
class JobManager:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        # State managed by the actor
        self.jobs = {}  # {job_id: job_info}
        self.tasks = {}  # {task_id: task_info}
        self._job_queue = []  # List of job_ids waiting to be processed serially
        self._current_job_id = None  # ID of the job currently being processed
        self._running_task_futures = (
            {}
        )  # {task_id: ray.ObjectRef} - tasks launched in the current job

        # Concurrency and monitoring
        self._monitor_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._monitor_thread = threading.Thread(target=self._run_monitor)
        self._monitor_thread.daemon = (
            True  # Allow main process to exit even if thread is running
        )

        self._load_checkpoint()  # Attempt to load state on startup
        self._monitor_thread.start()  # Start the monitoring thread

    def log(self, msg):
        logger.info(msg)

    def _save_checkpoint(self):
        """Saves the current actor state to a file."""
        try:
            # Only save the core state needed for recovery
            state_to_save = {
                "jobs": self.jobs,
                "tasks": self.tasks,
                "_job_queue": self._job_queue,
                "_current_job_id": self._current_job_id,
                # Don't save _running_task_futures as they are specific to the current Ray session/actor instance
            }
            with open(self.checkpoint_path, "wb") as f:
                pickle.dump(state_to_save, f)
            logger.debug("Checkpoint saved.")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}", exc_info=True)

    def _load_checkpoint(self):
        """Loads state from a checkpoint file if it exists."""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "rb") as f:
                    state = pickle.load(f)
                # Restore state
                self.jobs = state.get("jobs", {})
                self.tasks = state.get("tasks", {})
                self._job_queue = state.get("_job_queue", [])
                self._current_job_id = state.get("_current_job_id", None)

                # Rebuild running tasks state if current_job exists (best effort)
                # Note: We lose the actual ObjectRefs, so any tasks that were
                # RUNNING at the time of the crash will be considered PENDING again
                # and potentially retried, which is safer than assuming they succeeded.
                if self._current_job_id and self._current_job_id in self.jobs:
                    logger.warning(
                        f"[{self._current_job_id}] Restored from checkpoint."
                    )
                    # Reset running state and mark active tasks for retry
                    for task_id in self.jobs[self._current_job_id]["task_ids"]:
                        task_info = self.tasks.get(task_id)
                        if task_info and task_info["status"] in [
                            TaskStatus.RUNNING,
                        ]:
                            logger.warning(
                                f"[{self._current_job_id}] Task {task_id} was running. Resetting to retrying."
                            )
                            # Increment retries *here* because this counts as an attempt
                            task_info["retries"] += 1
                            if task_info["retries"] >= MAX_RETRIES:
                                task_info["status"] = TaskStatus.FAILED
                                task_info["result"] = (
                                    {
                                        "success": False,
                                        "error": "Task failed due to actor restart during execution.",
                                    },
                                )
                                task_info["end_time"] = time.time()
                                logger.error(
                                    f"[{self._current_job_id}] Task {task_id} failed permanently after actor restart."
                                )
                            else:
                                task_info["status"] = TaskStatus.RETRYING
                                task_info["start_time"] = None
                                task_info["end_time"] = None

                logger.info(f"State loaded from checkpoint at {self.checkpoint_path}.")
            except Exception as e:
                logger.error(
                    f"Failed to load checkpoint from {self.checkpoint_path}: {e}",
                    exc_info=True,
                )
                # Decide whether to raise or continue with empty state; continuing is safer for service availability
                self.jobs = {}
                self.tasks = {}
                self._job_queue = []
                self._current_job_id = None
        else:
            logger.info("No checkpoint found. Starting with empty state.")

    def _update_job_status(self, job_id):
        """Updates the overall job status based on its tasks."""
        job_info = self.jobs.get(job_id)
        if not job_info:
            logger.warning(f"Attempted to update status for non-existent job {job_id}")
            return

        task_ids = job_info.get("task_ids", [])
        status_counts: tp.Dict[TaskStatus, int] = defaultdict(int)
        for task_id in task_ids:
            task_info = self.tasks.get(task_id)
            status = task_info["status"] if task_info else TaskStatus.PENDING
            status_counts[status] += 1

        # Determine job status
        if (
            status_counts[TaskStatus.RUNNING]
            + status_counts[TaskStatus.PENDING]
            + status_counts[TaskStatus.RETRYING]
            > 0
        ):
            job_info["status"] = JobStatus.RUNNING
        elif status_counts[TaskStatus.FAILED] > 0:
            job_info["status"] = JobStatus.FAILED
            logger.info(f"Job {job_id} status updated to {JobStatus.FAILED}")
        elif status_counts[TaskStatus.SUCCEEDED] >= len(task_ids):
            job_info["status"] = JobStatus.COMPLETED
            logger.info(f"Job {job_id} status updated to {JobStatus.COMPLETED}")
        else:
            logger.warning(
                f"Job {job_id} in indeterminate state. Task counts: {dict(status_counts)}. Defaulting to RUNNING"
            )
            job_info["status"] = JobStatus.RUNNING

    def submit_job(self, job_definition):
        """
        Submits a new job with a list of task definitions and concurrency limit.
        """
        job_id = job_definition["job_id"]
        task_definitions = job_definition["task_definitions"]
        with self._monitor_lock:
            if job_id in self.jobs:
                raise JobAlreadyExist(
                    f"Job ID {job_id} already exists. Ignoring submission."
                )

            logger.info(f"Submitting job {job_id} with {len(task_definitions)} tasks.")

            job_applications = job_definition["applications"]
            job_info = {
                "job_id": job_id,
                "job_definition": job_definition,
                "status": JobStatus.SUBMITTED,
                "message": "",  # message for failed job
                "submit_time": time.time(),
                "task_ids": [],
                "app_status": ["NOT_STARTED" for _ in range(len(job_applications))],
                "app_status_check_attemps": 0,
            }
            self.jobs[job_id] = job_info

            for i, task_def in enumerate(task_definitions):
                task_id = generate_task_id(job_id, i)
                job_info["task_ids"].append(task_id)
                task_info = {
                    "task_id": task_id,
                    "job_id": job_id,
                    "index": i,
                    "status": TaskStatus.PENDING,
                    "retries": 0,
                    "submit_time": time.time(),
                    "start_time": None,  # When the task *sequence* (deploy/check/run) starts
                    "end_time": None,
                    "result": None,  # Stores the final (bool, data) tuple
                }
                self.tasks[task_id] = task_info

            self._job_queue.append(job_id)  # Add job to the processing queue
            self._save_checkpoint()  # Save state after adding a new job
            logger.info(
                f"Job {job_id} added to queue. Queue size: {len(self._job_queue)}"
            )

        return job_id  # Return job ID immediately

    def _run_monitor(self):
        """Main loop for the monitoring thread that processes jobs serially and manages task concurrency."""
        logger.info("Monitor thread started.")
        while not self._stop_event.is_set():
            try:
                with self._monitor_lock:
                    self._start_next_job_if_needed()
                    if self._current_job_id:
                        self._process_current_job()
            except Exception as e:
                logger.error(f"Exception in monitor thread: {e}", exc_info=True)
            self._stop_event.wait(JOB_MONITOR_INTERVAL)
        logger.info("Monitor thread stopped.")

    def _start_next_job_if_needed(self):
        """Start the next job from the queue if no job is currently active."""
        if self._current_job_id is None and self._job_queue:
            self._current_job_id = self._job_queue.pop(0)
            self.jobs[self._current_job_id]["status"] = JobStatus.DEPLOYING
            logger.info(f"Monitor started processing job: {self._current_job_id}")
            self._save_checkpoint()

    def _process_current_job(self):
        """Process the current active job based on its status."""
        job_info = self.jobs[self._current_job_id]
        status = job_info["status"]

        if status == JobStatus.DEPLOYING:
            self._handle_deploying_job(job_info)
        elif status == JobStatus.STATUSCHECKING:
            self._handle_health_checking_job(job_info)
        elif status == JobStatus.RUNNING:
            self._handle_running_job(job_info)
        elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            self._handle_finished_job(job_info)
        self._save_checkpoint()

    def _handle_deploying_job(self, job_info):
        """Deploy applications for the job."""
        job_id = job_info["job_id"]
        job_def = job_info["job_definition"]
        deploy_func = job_def["deploy_applications"]
        applications = job_def["applications"]

        logger.info(f"[{job_id}] Starting deployment...")
        try:
            deployment_result = deploy_func(applications)
            logger.info(f"[{job_id}] Deployment finished. Result: {deployment_result}")
            job_info["status"] = JobStatus.STATUSCHECKING
        except Exception as e:
            logger.error(f"[{job_id}] Deployment failed: {e}", exc_info=True)
            job_info["status"] = JobStatus.FAILED
            job_info["message"] = "deployment failed"

    def _handle_health_checking_job(self, job_info):
        """Check the health status of deployed applications."""
        job_id = job_info["job_id"]
        job_def = job_info["job_definition"]
        check_status_func = job_def["check_status"]
        applications = job_def["applications"]
        timeout = job_def["timeout"]

        attempt = job_info["app_status_check_attemps"]
        max_status_check_attempts = int(timeout // STATUS_CHECK_INTERVAL)
        if not applications:
            job_info["status"] = JobStatus.RUNNING

        for i, app in enumerate(applications):
            status = job_info["app_status"][i]
            if not status_is_pending(status):
                continue

            status = check_status_func(app)
            logger.info(f"[{job_id}] Health check attempt {attempt}: {app} {status}")
            job_info["app_status"][i] = status

            if status_is_failure(status):
                job_info["status"] = JobStatus.FAILED
                job_info["result"] = f"app {app} has bad status {status}"
            elif status_is_success(status) and i == len(applications) - 1:
                logger.info(f"[{job_id}] all apps are ready")
                job_info["status"] = JobStatus.RUNNING
            elif status_is_pending(status) and attempt == max_status_check_attempts:
                job_info["status"] = JobStatus.FAILED
                job_info["result"] = f"Too many status checking attempts {attempt}"
            job_info["app_status_check_attemps"] += 1

            break  # only do one app

    def _handle_running_job(self, job_info):
        """Process tasks for a running job."""
        job_id = job_info["job_id"]
        job_def = job_info["job_definition"]
        timeout = job_def["timeout"]

        self._process_completed_tasks()
        self._handle_timed_out_tasks(job_id, timeout)
        self._schedule_new_tasks(job_info)
        self._update_job_status(job_id)

    def _process_completed_tasks(self):
        """Check and process any completed tasks."""
        ready_futures, _ = ray.wait(
            list(self._running_task_futures.values()), timeout=0
        )

        for task_id, future in list(self._running_task_futures.items()):
            if future not in ready_futures:
                continue

            self._running_task_futures.pop(task_id)
            try:
                result_data = ray.get(future)
                self._process_task_completion(task_id, result_data)
            except Exception as e:
                logger.error(f"[{task_id}] Error getting result: {e}", exc_info=True)
                self._process_task_completion(
                    task_id,
                    {"success": False, "error": f"Failed to retrieve task result: {e}"},
                )

    def _handle_timed_out_tasks(self, job_id, timeout):
        """Check for and handle any timed out tasks."""
        now = time.time()
        for task_id, task_info in list(self.tasks.items()):
            # Skip tasks that don't meet timeout conditions
            if (
                task_info["job_id"] != job_id
                or task_info["status"] != TaskStatus.RUNNING
                or task_info["start_time"] is None
                or (now - task_info["start_time"]) <= timeout
            ):
                continue

            logger.warning(
                f"[{task_id}] Task timed out after {now - task_info['start_time']:.2f}s (limit: {timeout}s)"
            )

            # Kill the task if it has a running future
            future = self._running_task_futures.pop(task_id, None)
            if future:
                try:
                    ray.cancel(future)
                    logger.info(f"[{task_id}] Ray task killed due to timeout")
                except Exception as e:
                    logger.error(
                        f"[{task_id}] Failed to kill Ray task: {e}", exc_info=True
                    )

            # Process as a failure
            self._process_task_completion(
                task_id,
                {
                    "success": False,
                    "error": f"Task timed out after exceeding {timeout} seconds",
                },
            )

    def _schedule_new_tasks(self, job_info):
        task_ids = job_info["task_ids"]
        job_def = job_info["job_definition"]
        max_concurrent = job_def["max_concurrent_tasks"]

        """Schedule new tasks if concurrency allows."""
        available_slots = max_concurrent - len(self._running_task_futures)
        if available_slots <= 0:
            return

        pending_task_ids = [
            tid
            for tid in task_ids
            if self.tasks[tid]["status"] in [TaskStatus.PENDING, TaskStatus.RETRYING]
        ]

        for task_id in pending_task_ids[:available_slots]:
            self._launch_task(task_id, job_def)

    def _handle_finished_job(self, job_info):
        """Clean up after a completed or failed job."""
        job_id = job_info["job_id"]
        job_def = job_info["job_definition"]
        cleanup_func = job_def["cleanup_applications"]
        applications = job_def["applications"] + [
            app
            for task_def in job_def["task_definitions"]
            for app in task_def["applications"]
        ]

        if cleanup_func:
            logger.info(f"[{job_id}] Starting cleanup...")
            try:
                cleanup_results = cleanup_func(applications)
                logger.info(
                    f"[{job_id}] Cleanup finished successfully removed {cleanup_results}"
                )
            except Exception as e:
                logger.error(f"[{job_id}] Cleanup failed: {e}", exc_info=True)

        self._current_job_id = None  # Move to the next job
        self._running_task_futures = {}

    def _launch_task(self, task_id, job_def):
        """Launch a Ray task for the given task_id. Assumes monitor lock is held."""
        task_info = self.tasks.get(task_id)
        if not task_info:
            logger.warning(
                f"Cannot launch task {task_id}: not in PENDING/RETRYING state"
            )
            return

        job_id = task_info["job_id"]

        # Get task configuration
        task_index = task_info["index"]
        timeout = job_def["timeout"]
        task_def = job_def["task_definitions"][task_index]

        # Extract task parameters
        user_func = task_def["func"]
        args = task_def["args"]
        kwargs = task_def["kwargs"]
        applications = task_def["applications"]
        resources = task_def["resources"]

        # Update task status
        task_info["status"] = TaskStatus.RUNNING
        task_info["start_time"] = time.time()

        logger.info(
            f"Launching task {task_id} (Attempt {task_info['retries'] + 1}) with resources: {resources}"
        )

        # Build Ray options
        task_options = {
            "num_cpus": resources.get("CPU", 0),
            "num_gpus": resources.get("GPU", 0),
        }

        # Launch remote task
        future = _execute_task_sequence.options(**task_options).remote(
            user_func,
            args,
            kwargs,
            applications,
            job_def["deploy_applications"],
            job_def["check_status"],
            job_def["cleanup_applications"],
            timeout,
            task_id,
        )
        self._running_task_futures[task_id] = future

    def _process_task_completion(self, task_id, result_data):
        """
        Internal method called by the monitor thread when a Ray task future is ready.
        Processes the result, updates state, handles retries, and saves checkpoint.
        Assumes monitor lock is held when called.
        """
        task_info = self.tasks.get(task_id)
        if not task_info:
            logger.warning(
                f"Received completion report for unknown/stale task {task_id}. Ignoring."
            )
            # Should ideally not happen if called from _run_monitor after ray.wait
            return

        job_id = task_info["job_id"]
        task_info["end_time"] = time.time()
        success = result_data.get("success", False)
        if success:
            logger.info(f"[{task_id}] Succeeded.")
            task_info["status"] = TaskStatus.SUCCEEDED
            task_info["result"] = result_data
        else:
            logger.warning(f"[{task_id}] Failed on attempt {task_info['retries'] + 1}.")
            task_info["retries"] += 1
            task_info["result"] = result_data  # Store failure reason

            if task_info["retries"] < MAX_RETRIES:
                logger.info(
                    f"[{task_id}] Retrying task (Attempt {task_info['retries'] + 1}/{MAX_RETRIES})."
                )
                task_info["status"] = (
                    TaskStatus.RETRYING
                )  # Mark for rescheduling by monitor
                task_info["start_time"] = None  # Reset start time for the next attempt
                task_info["end_time"] = None
                # The monitor loop will find this task in RETRYING/PENDING status and reschedule it
            else:
                logger.error(
                    f"[{task_id}] Failed permanently after {MAX_RETRIES} attempts."
                )
                task_info["status"] = TaskStatus.FAILED

    def get_job_status(self, job_id):
        """Gets the status and progress of a job."""
        with self._monitor_lock:
            if job_id not in self.jobs:
                raise JobNotFound(f"Job with ID '{job_id}' not found.")

            job_info = self.jobs[job_id]
            task_ids = job_info["task_ids"]
            num_tasks = len(task_ids)

            status_counts: tp.Dict[TaskStatus, int] = defaultdict(int)
            for task_id in task_ids:
                task_info = self.tasks.get(task_id)
                if task_info:
                    status_counts[task_info["status"]] += 1
                else:
                    # Should not happen, but handle defensively
                    status_counts[TaskStatus.PENDING] += 1

            return {
                "job_id": job_id,
                "status": job_info["status"].value,  # Return enum value (string)
                "message": job_info["message"],
                "total_tasks": num_tasks,
                "tasks_succeeded": status_counts[TaskStatus.SUCCEEDED],
                "tasks_failed": status_counts[TaskStatus.FAILED],
                "submit_time": job_info["submit_time"],
                "tasks_in_queue": len(
                    [
                        tid
                        for tid in task_ids
                        if self.tasks[tid]["status"]
                        in [TaskStatus.PENDING, TaskStatus.RETRYING]
                    ]
                ),
                "tasks_active": len(
                    set(self._running_task_futures.keys()) & set(task_ids)
                ),  # Number of Ray futures launched for the current job
            }

    def get_job_results(self, job_id):
        """Gets the results for all tasks within a job."""
        with self._monitor_lock:
            if job_id not in self.jobs:
                raise JobNotFound(f"Job with ID '{job_id}' not found.")

            job_info = self.jobs[job_id]
            # Allow fetching results even if running, but note incomplete tasks
            job_status = job_info["status"]
            if job_status not in [JobStatus.COMPLETED, JobStatus.FAILED]:
                logger.warning(
                    f"Job {job_id} is still {job_status.value}. Results may be incomplete."
                )

            results = {}
            for index, task_id in enumerate(job_info["task_ids"]):
                task_info = self.tasks.get(task_id)
                task_name = job_info["job_definition"]["task_definitions"][index][
                    "task_id"
                ]
                if task_info and task_info["status"] in [
                    TaskStatus.SUCCEEDED,
                    TaskStatus.FAILED,
                ]:
                    result = task_info["result"].copy()
                else:
                    result = {
                        "success": False,
                        "error": f"Task not completed Status",
                    }
                # Task not finished or info missing
                status = (
                    task_info["status"].value
                    if task_info and task_info["status"]
                    else "UNKNOWN"
                )
                if task_info and task_info.get("start_time"):
                    if task_info.get("end_time"):
                        running_time = int(
                            task_info["end_time"] - task_info["start_time"]
                        )
                    else:
                        running_time = int(time.time() - task_info["start_time"])
                else:
                    running_time = 0
                result.update(
                    {
                        "status": status,
                        "running_time": running_time,
                    }
                )
                results[task_name] = result

            return results

    def get_all_job_ids(self):
        """Returns a list of all known job IDs."""
        with self._monitor_lock:
            # Include jobs in the queue and the current job
            all_ids = list(self.jobs.keys())
            return all_ids

    def delete_job(self, job_id):
        """remove one job"""
        with self._monitor_lock:
            if job_id not in self.jobs:
                raise JobNotFound(f"Job with ID '{job_id}' not found.")

            job_info = self.jobs.pop(job_id)
            if job_id in self._job_queue:
                self._job_queue.remove(job_id)

            job_status = job_info["status"]
            if job_status == JobStatus.RUNNING:
                logger.warning(
                    f"Job {job_id} is still {job_status.value}. kill anyway."
                )

            for task_id in job_info["task_ids"]:
                task_info = self.tasks.pop(task_id)
                future = self._running_task_futures.pop(task_id, None)
                if future:
                    try:
                        ray.cancel(future)
                        logger.info(f"[{task_id}] Ray task killed")
                    except Exception as e:
                        logger.error(
                            f"[{task_id}] Failed to kill Ray task: {e}", exc_info=True
                        )

            if self._current_job_id == job_id:
                self._handle_finished_job(job_info)
            self._save_checkpoint()

    def stop(self):
        """Gracefully shuts down the monitor thread."""
        logger.info("JobManager actor shutting down monitor thread.")
        self._stop_event.set()
        self._monitor_thread.join(timeout=5)  # Wait for thread to finish (with timeout)
        if self._monitor_thread.is_alive():
            logger.warning("Monitor thread did not join gracefully.")
        else:
            logger.info("Monitor thread joined.")

        with self._monitor_lock:
            # Attempt to kill any remaining running Ray tasks for the current job
            if self._current_job_id and self._running_task_futures:
                logger.warning(
                    f"Killing {len(self._running_task_futures)} running tasks for job {self._current_job_id} during shutdown."
                )
                for task_id, future in list(self._running_task_futures.items()):
                    try:
                        ray.cancel(future)
                        logger.info(f"Killed task {task_id}.")
                    except Exception as e:
                        logger.error(
                            f"Failed to kill task {task_id} during shutdown: {e}"
                        )
                self._running_task_futures = {}  # Clear the list

            self._save_checkpoint()  # Save final state before exiting

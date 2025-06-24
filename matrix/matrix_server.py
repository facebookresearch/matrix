# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import getpass
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import fire
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

import matrix
from matrix import Cli
from matrix.job.eval_utils import *
from matrix.job.job_api import JobApi
from matrix.utils.os import download_s3_dir, find_free_ports, is_port_available
from matrix.utils.ray import get_ray_address

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Job API Service", description="HTTP Service for Matrix Job API")

global_cluster_id: str | None = None
global_matrix_dir: str | None = None


# Models for request/response
class StartClusterRequest(BaseModel):
    """Request model for starting a cluster"""

    add_workers: int = Field(0, description="Number of workers to add")
    slurm: Optional[Dict[str, Union[str, int]]] = Field(
        None, description="Slurm configuration"
    )
    local: Optional[Dict[str, Union[str, int]]] = Field(
        None, description="Local configuration"
    )
    enable_grafana: bool = Field(False, description="Enable Grafana")
    force_new_head: bool = Field(False, description="Force creation of new head node")


class StartClusterResponse(BaseModel):
    """Response model for starting a cluster"""

    workers: List[str]
    cluster_info: Dict[str, Any]


class DeployApplicationsRequest(BaseModel):
    """Request model for deploying applications"""

    action: str = Field(default="replace", description="Deployment action")
    applications: Optional[List[Dict[str, Union[str, int]]]] = Field(
        None, description="Application configurations"
    )
    yaml_config: Optional[str] = Field(None, description="YAML configuration string")


class TaskDefinition(BaseModel):
    task_id: Optional[str] = None
    func: str  # Function name as string, will be converted to callable
    args: tuple = ()
    kwargs: dict = {}
    resources: dict = {"CPU": 1}
    applications: List = []


# Checkpoint evaluation models
class CheckpointEvalRequest(BaseModel):
    checkpoint_dir: str
    eval_save_dir: str
    min_replica: int = 8
    max_replica: int = 64
    thinking: bool = True
    job_id: Optional[str] = None
    matrix_dir: Optional[str] = None
    benchmarks: Optional[List[str]] = None
    num_seeds: Optional[int] = None
    max_concurrent_tasks: int = 8
    timeout: int = 36000
    model_size: str = "8B"
    tokenizer: str = "meta-llama/Llama-3.1-8B-Instruct"
    use_ray_data: bool = True
    sampling_params: Dict[str, Any] | None = None
    skip_generation: bool = False


class BenchmarkStatus(BaseModel):
    successes: int
    failures: int
    pending: int
    metric_values: List[float]
    metric_avg: float
    metric_stderr: float


class CheckpointEvalMetrics(BaseModel):
    job_id: str
    status: str
    results: Optional[Dict[str, Dict[str, Any]]] = None
    benchmark_stats: Optional[Dict[str, BenchmarkStatus]] = None


class JobDefinition(BaseModel):
    job_id: Optional[str] = None
    max_concurrent_tasks: int = 1
    timeout: int = 36000
    applications: List = []
    task_definitions: List[TaskDefinition]
    deploy_applications: Optional[str] = None
    check_status: Optional[str] = None
    cleanup_applications: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    status: str
    message: str
    total_tasks: int
    tasks_succeeded: int
    tasks_failed: int
    tasks_in_queue: int
    tasks_active: int
    submit_time: Optional[float] = None


class JobResult(BaseModel):
    task_results: Dict[str, Dict[str, Any]]


class JobIdResponse(BaseModel):
    job_id: str


class JobIdsResponse(BaseModel):
    job_ids: List[str]


# Dependency for JobApi instance
def get_matrix_cli():
    cli = matrix.Cli(cluster_id=global_cluster_id, matrix_dir=global_matrix_dir)
    return cli


def get_job_api():
    cli = matrix.Cli(cluster_id=global_cluster_id, matrix_dir=global_matrix_dir)
    return cli.job


@app.post("/start-cluster", response_model=StartClusterResponse)
async def start_cluster_endpoint(
    request: StartClusterRequest, cli: Cli = Depends(get_matrix_cli)
):
    """Create a cluster"""
    try:
        return cli.start_cluster(
            request.add_workers,
            request.slurm,
            request.local,
            request.enable_grafana,
            request.force_new_head,
        )
    except Exception as e:
        logger.error(f"Error start cluster: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop-cluster")
async def stop_cluster_endpoint(cli: Cli = Depends(get_matrix_cli)):
    """
    Stop the currently running cluster.

    Returns:
        No content on success
    """
    try:
        # Assuming your cluster manager has a stop_cluster method
        cli.stop_cluster()
        return {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop cluster: {str(e)}")


@app.get("/status")
async def status_endpoint(cli: Cli = Depends(get_matrix_cli)):
    try:
        return {"status": cli.status()}
    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deploy-applications")
async def deploy_applications_endpoint(
    request: DeployApplicationsRequest, cli: Cli = Depends(get_matrix_cli)
):
    try:
        app_names = cli.deploy_applications(
            request.action,
            request.applications,
            request.yaml_config,
        )
        return {"applications": app_names}
    except Exception as e:
        logger.error(f"Error deploy applications: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/app-status")
async def app_status(app_name: str, cli: Cli = Depends(get_matrix_cli)):
    try:
        status = cli.app.app_status(app_name)
        return {"name": app_name, "app_status": status}
    except Exception as e:
        logger.error(f"Error app status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/jobs", response_model=JobIdResponse)
async def submit_job(
    job_definition: JobDefinition, job_api: JobApi = Depends(get_job_api)
):
    """Submit a new job to the JobManager"""
    try:
        # Convert Pydantic model to dict
        job_def_dict = job_definition.dict()

        # Submit the job
        job_id = job_api.submit(job_def_dict)
        return {"job_id": job_id}
    except Exception as e:
        logger.error(f"Error submitting job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/status", response_model=JobStatus)
async def get_job_status(job_id: str, job_api: JobApi = Depends(get_job_api)):
    """Get the status of a job"""
    try:
        status = job_api.status(job_id)
        return status
    except Exception as e:
        logger.error(f"Error getting job status: {e}", exc_info=True)
        raise HTTPException(
            status_code=404 if "JobNotFound" in str(e) else 500, detail=str(e)
        )


@app.get("/jobs/{job_id}/results", response_model=JobResult)
async def get_job_results(job_id: str, job_api: JobApi = Depends(get_job_api)):
    """Get the results of a job"""
    try:
        results = job_api.get_results(job_id)
        return {"task_results": results}
    except Exception as e:
        logger.error(f"Error getting job results: {e}", exc_info=True)
        raise HTTPException(
            status_code=404 if "JobNotFound" in str(e) else 500, detail=str(e)
        )


@app.get("/jobs", response_model=JobIdsResponse)
async def list_jobs(job_api: JobApi = Depends(get_job_api)):
    """List all job IDs"""
    try:
        job_ids = job_api.list()
        return {"job_ids": job_ids}
    except Exception as e:
        logger.error(f"Error listing jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str, job_api: JobApi = Depends(get_job_api)):
    """Delete a job"""
    try:
        job_api.delete(job_id)
        return {"message": f"Job {job_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting job: {e}", exc_info=True)
        raise HTTPException(
            status_code=404 if "JobNotFound" in str(e) else 500, detail=str(e)
        )


@app.delete("/jobs")
async def clear_jobs(job_api: JobApi = Depends(get_job_api)):
    """Clear all jobs"""
    try:
        deleted_jobs = job_api.clear()
        return {
            "message": f"Deleted {len(deleted_jobs)} jobs",
            "deleted_jobs": deleted_jobs,
        }
    except Exception as e:
        logger.error(f"Error clearing jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/checkpoint-eval", response_model=CheckpointEvalMetrics)
async def evaluate_checkpoint(
    request: CheckpointEvalRequest,
    cli: Cli = Depends(get_matrix_cli),
    job_api: JobApi = Depends(get_job_api),
):
    """
    Start a checkpoint evaluation job and return the job ID.

    This endpoint creates and submits a job for evaluating a model checkpoint
    on various benchmarks.
    """
    if os.environ.get("CHECKPOINT_EVAL_SCRIPT") is None:
        raise HTTPException(
            status_code=400, detail="Need to set environment CHECKPOINT_EVAL_SCRIPT"
        )
    if os.environ.get("CHECKPOINT_EVAL_PYTHONPATH") is None:
        raise HTTPException(
            status_code=400, detail="Need to set environment CHECKPOINT_EVAL_PYTHONPATH"
        )

    try:
        checkpoint_dir = request.checkpoint_dir
        if checkpoint_dir.startswith("s3://"):
            cache_dir = os.environ.get(
                "MATRIX_CACHE", os.path.expanduser("~/.cache/matrix")
            )
            assert cli.cluster_id is not None
            cache_dir = os.path.join(cache_dir, cli.cluster_id, "models")
            s3_dir = checkpoint_dir
            logger.info(f"Download {s3_dir} under {cache_dir}")
            downloaded, dest_dir = await asyncio.to_thread(
                lambda: download_s3_dir(s3_dir, cache_dir, 3, "*rank_*.pt")
            )
            if not downloaded:
                raise ValueError(f"Can not read {s3_dir}")
            request.checkpoint_dir = dest_dir

        benchmarks = request.benchmarks or list(BENCHMARK_CONFIG.keys())

        if request.job_id:
            app_name = request.job_id.replace("/", "-")
        else:
            app_name = "-".join(request.checkpoint_dir.split("/")[-3:])
        job_id = request.job_id or app_name
        if request.skip_generation:
            job_ids = job_api.list()
            if job_id in job_ids:
                logger.info(f"Delete the existing job: {job_id}")
                job_api.delete(job_id)

        task_definitions = []
        cluster_info = cli.cluster.cluster_info()
        assert cluster_info is not None
        assert global_cluster_id is not None and global_matrix_dir is not None
        for benchmark in benchmarks:
            num_seeds = request.num_seeds or DEFAULT_NUM_SEEDS[benchmark]
            start_seed = 1 if num_seeds > 1 else 42
            for seed in range(start_seed, start_seed + num_seeds):
                env, command = run_eval_script(
                    os.environ["CHECKPOINT_EVAL_PYTHONPATH"],
                    os.environ["CHECKPOINT_EVAL_SCRIPT"],
                    os.path.join(request.eval_save_dir, benchmark, app_name),
                    request.checkpoint_dir,
                    benchmark,
                    seed,
                    request.thinking,
                    global_cluster_id,
                    global_matrix_dir,
                    app_name,
                    request.use_ray_data,
                    get_ray_address(cluster_info),
                    request.tokenizer,
                    request.sampling_params,
                    request.skip_generation,
                )
                task_definitions.append(
                    {
                        "task_id": f"{benchmark}_seed{seed}",
                        "func": "matrix.utils.os.run_and_stream",
                        "kwargs": {
                            "command": command,
                            "blocking": True,
                            "env": env,
                            "return_stdout_lines": 100,
                            "skip_logging": " INFO:",
                        },
                    }
                )

        job_def = {
            "job_id": job_id,
            "applications": (
                [
                    {
                        "name": app_name,
                        "model_name": request.checkpoint_dir,
                        "model_size": request.model_size,
                        "tokenizer": request.tokenizer,
                        "use_grpc": "true",
                        "min_replica": request.min_replica,
                        "max_replica": request.max_replica,
                    }
                ]
                if not request.use_ray_data
                else []
            ),
            "task_definitions": task_definitions,
            "timeout": request.timeout,
            "max_concurrent_tasks": request.max_concurrent_tasks,
        }
        logger.info(f"Submitting checkpoint evaluation job: {job_def}")

        job_id = job_api.submit(job_def)
        logger.info(f"Submitted checkpoint evaluation job: {job_id}")

        return {"job_id": job_id, "status": "SUBMITTED"}

    except Exception as e:
        logger.error(f"Error starting checkpoint evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/checkpoint-eval/{job_id}/metrics", response_model=CheckpointEvalMetrics)
async def get_checkpoint_eval_metrics(
    job_id: str, job_api: JobApi = Depends(get_job_api)
):
    """
    Get the status of a checkpoint evaluation job.

    If the job is completed, also returns the benchmark statistics.
    """
    try:
        status = job_api.status(job_id)
        response = {"job_id": job_id, "status": status["status"]}
        results = job_api.get_results(job_id)
        response["results"] = results

        benchmark_stats = extract_benchmark_data(
            results, r"(?:Acc Score|Exact Match Score): ([\d.]+)"
        )
        response["benchmark_stats"] = benchmark_stats

        return response

    except Exception as e:
        logger.error(f"Error getting checkpoint evaluation status: {e}", exc_info=True)
        raise HTTPException(
            status_code=404 if "JobNotFound" in str(e) else 500, detail=str(e)
        )


@app.post("/job-manager/start")
async def start_job_manager(job_api: JobApi = Depends(get_job_api)):
    """Start the JobManager actor"""
    try:
        job_api.start()
        return {"message": "JobManager started successfully"}
    except Exception as e:
        logger.error(f"Error starting JobManager: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/job-manager/stop")
async def stop_job_manager(job_api: JobApi = Depends(get_job_api)):
    """Stop the JobManager actor"""
    try:
        job_api.stop()
        return {"message": "JobManager stopped successfully"}
    except Exception as e:
        logger.error(f"Error stopping JobManager: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


def main(
    cluster_id: str | None = None,
    matrix_dir: str | None = None,
    port: int | None = 6289,
):
    global global_cluster_id, global_matrix_dir
    cluster_id = (
        cluster_id or os.getenv("MATRIX_CLUSTER_ID") or (getpass.getuser() + "_cluster")
    )

    global_cluster_id = cluster_id
    if matrix_dir is not None:
        global_matrix_dir = matrix_dir
    else:
        global_matrix_dir = os.path.expanduser("~/.matrix")
    if port is None or not is_port_available(port):
        port = find_free_ports(1)[0]
    assert port is not None
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    fire.Fire(main)

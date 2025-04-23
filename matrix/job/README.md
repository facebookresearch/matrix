# Matrix Job Manager

A Python package for submitting and managing tasks using Ray, with support for concurrency limits, retries, timeouts, and application lifecycle handling.

## Features

### Job Submission and Monitoring  
- Jobs consist of multiple tasks and are submitted with a unique `job_id`.
- A background thread processes jobs one at a time.
- Job and task states are checkpointed to support resuming after restarts.

### Task Execution  
- A concurrency limit (`k`) can be set to control how many tasks run in parallel.
- Failed tasks can be retried up to three times.
- Job define a timeout; tasks exceeding this will be terminated.

### Deployment Lifecycle  
- Tasks deploy applications using `deploy_applications`.
- Deployment status is checked using `check_status`.
- Cleanup is performed after task completion or cancellation.

### Thread Safety  
- Shared job state is protected with locks held briefly to avoid blocking other operations.

### Logging  
- Logs are available in the Ray dashboard under the actor name `JobManager`.

## Task Result Format

```python
{
    "success": bool,
    ... (other user-defined fields)
}
```

## Examples

### Run a User-Defined Function

```bash
matrix job submit "{'task_definitions': [{'func': 'matrix.job.job_utils.echo', 'args': ['hello']}]}"
```

### Run a Custom script

For checkpoint evaluation, see the [eval_checkpoints.py](../client/eval_checkpoints.py) for detials.

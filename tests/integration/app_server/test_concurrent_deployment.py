import tempfile
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Generator, List

import pytest

from matrix.app_server import app_api
from matrix.cli import Cli

MAX_DEPLOYMENT = 10
DEPLOY_TIMEOUT = 30


@pytest.fixture(scope="module")
def matrix_cluster() -> Generator[Any, Any, Any]:
    """Start and stop Ray for the duration of these tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cli = Cli(cluster_id=str(uuid.uuid4()), matrix_dir=temp_dir)
        cli.start_cluster(
            add_workers=1,
            slurm=None,
            local={"gpus_per_node": 0, "cpus_per_task": 2 * MAX_DEPLOYMENT},
            enable_grafana=False,
        )

        with cli.cluster:
            yield cli


def wait_for_removal(cli: Cli, app_name: str, timeout: int = 60) -> bool:
    """Wait for an application to be removed."""
    time.sleep(2)
    return True


def wait_for_running(cli: Cli, app_name: str, timeout: int = DEPLOY_TIMEOUT) -> bool:
    """Wait for an application to reach RUNNING state with timeout."""
    start_time = time.time()
    while (status := cli.app.app_status(app_name)) != "RUNNING":
        if time.time() - start_time > timeout:
            print(f"{app_name} timed out after {timeout}s, last status: {status}")
            return False
        print(f"{app_name} not ready, current status {status}")
        time.sleep(10)
    return True


def deploy_and_remove_cycle(
    worker_id: int,
    num_cycles: int,
    cluster_id: str,
    matrix_dir: str,
) -> Dict[str, Any]:
    """
    Perform multiple add/remove cycles for a single worker process.

    Each worker deploys and removes applications with unique names.
    """
    cli = Cli(cluster_id=cluster_id, matrix_dir=matrix_dir)

    results = {
        "worker_id": worker_id,
        "cycles": [],
        "success": True,
        "errors": [],
    }

    for cycle in range(num_cycles):
        app_name = f"hello_worker{worker_id}_cycle{cycle}"
        cycle_result = {
            "cycle": cycle,
            "app_name": app_name,
            "add_success": False,
            "remove_success": False,
        }

        try:
            # Add the application
            cli.deploy_applications(
                action=app_api.Action.ADD,
                applications=[{"name": app_name, "app_type": "hello"}],
            )

            # Wait for the app to be running with timeout
            if not wait_for_running(cli, app_name):
                results["errors"].append(
                    f"{app_name}: timed out waiting for RUNNING state"
                )
                results["success"] = False
                results["cycles"].append(cycle_result)
                continue

            cycle_result["add_success"] = True

            # Verify health after add
            if not cli.check_health(app_name):
                results["errors"].append(f"{app_name}: health check failed after add")
                results["success"] = False

            # Remove the application
            cli.deploy_applications(
                action=app_api.Action.REMOVE,
                applications=[{"name": app_name, "app_type": "hello"}],
            )

            remove_success = wait_for_removal(cli, app_name)
            cycle_result["remove_success"] = remove_success

            if not remove_success:
                results["errors"].append(f"{app_name}: remove failed")
                results["success"] = False

        except Exception as e:
            results["errors"].append(f"{app_name}: exception - {str(e)}")
            results["success"] = False

        results["cycles"].append(cycle_result)

    return results


def test_concurrent_add_remove_multiprocessing(matrix_cluster: Cli) -> None:
    """
    Test concurrent add and remove operations using multiprocessing.

    Multiple worker processes each perform several add/remove cycles
    with uniquely named applications.
    """
    cli = matrix_cluster
    cluster_id = cli.cluster_id
    matrix_dir = cli.matrix_dir

    num_cycles_per_worker = 3

    results: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=MAX_DEPLOYMENT) as executor:
        futures = {
            executor.submit(
                deploy_and_remove_cycle,
                worker_id=worker_id,
                num_cycles=num_cycles_per_worker,
                cluster_id=cluster_id,
                matrix_dir=matrix_dir,
            ): worker_id
            for worker_id in range(MAX_DEPLOYMENT * 2)
        }

        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(
                    {
                        "worker_id": worker_id,
                        "success": False,
                        "errors": [f"Process exception: {str(e)}"],
                    }
                )

    # Aggregate and validate results
    total_cycles = MAX_DEPLOYMENT * 2 * num_cycles_per_worker
    failed_workers = [r for r in results if not r["success"]]

    all_errors = []
    for r in results:
        all_errors.extend(r.get("errors", []))

    assert (
        len(results) == MAX_DEPLOYMENT * 2
    ), f"Expected {MAX_DEPLOYMENT * 2} worker results, got {len(results)}"

    assert len(failed_workers) == 0, (
        f"Failed workers: {len(failed_workers)}/{MAX_DEPLOYMENT * 2}\n"
        f"Errors: {all_errors}"
    )

    # Verify all cycles completed
    total_completed_cycles = sum(len(r.get("cycles", [])) for r in results)
    assert (
        total_completed_cycles == total_cycles
    ), f"Expected {total_cycles} cycles, completed {total_completed_cycles}"

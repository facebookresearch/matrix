import asyncio
import json
from multiprocessing import Process, Queue
from typing import Any, Callable, Iterator, Optional

from tqdm import tqdm

import matrix.client.query_llm as query_llm
from matrix import Cli
from matrix.app_server.app_api import AppApi


class LLMClient:

    def __init__(
        self,
        app_name: str,
        cluster_id: Optional[str] = None,
        matrix_dir: Optional[str] = None,
    ):

        self.cluster = Cli(cluster_id=cluster_id, matrix_dir=matrix_dir).cluster
        app = AppApi(
            self.cluster.matrix_dir / self.cluster.cluster_id,
            self.cluster.cluster_info(),
        )
        self.app_name = app_name
        self.metadata = app.get_app_metadata(app_name)
        self.model_name = self.metadata["model_name"]
        self.endpoint_cache = self.metadata["endpoints"]["updater"]
        self.n_request_per_process = 1000

    async def get_response(self, data: dict[str, Any], params: dict) -> dict[str, Any]:
        """
        - data must include either "messages" or "prompt", other fields will be passed to response, no usage
        - params refer to https://fburl.com/lclc30sg
        """
        return await query_llm.make_request(
            url="",
            app_name=self.app_name,
            model=self.model_name,
            data=data,
            endpoint_cache=self.endpoint_cache,
            **(params),
        )

    def inference_process_wrapper(self, proc_id, params, request_queue, response_queue):
        batch_tasks = set()  # type: ignore

        async def populate_output_queue():
            nonlocal batch_tasks
            if batch_tasks:
                completed, batch_tasks = await asyncio.wait(
                    batch_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for completed_task in completed:
                    result = await completed_task
                    await asyncio.to_thread(response_queue.put, result)

        async def async_inference_process():
            nonlocal batch_tasks
            while True:
                data = await asyncio.to_thread(
                    request_queue.get
                )  # to make sure queue.get is blocking in a async function
                if data is None:
                    await asyncio.to_thread(request_queue.put, None)
                    break

                batch_tasks.add(asyncio.create_task(self.get_response(data, params)))
                if len(batch_tasks) >= self.n_request_per_process:
                    await populate_output_queue()

            while batch_tasks:
                await populate_output_queue()

        asyncio.run(async_inference_process())

    def postprocess(self, response_queue, output_filepath):
        pbar = tqdm()
        with open(output_filepath, "w") as file:
            while True:
                item = response_queue.get()
                if item is None:
                    break
                file.write(json.dumps(item) + "\n")
                pbar.update(1)
        pbar.close()

    def multiprocess_inference(
        self,
        data_loader: Iterator[dict[str, Any]],
        task_params: dict[str, Any],
        postproc_func: Callable = None,
        output_filepath: str = None,
        n_process=1,
        n_request_per_process=1000,
    ):
        self.n_request_per_process = n_request_per_process
        if postproc_func is None:
            postproc_func = self.postprocess
            assert (
                output_filepath is not None
            ), "Need to specify output file path if using the default post processing function."

        request_queue = Queue()  # type: ignore [var-annotated]
        response_queue = Queue()  # type: ignore [var-annotated]

        # Start consumer processes
        consumers = [
            Process(
                target=self.inference_process_wrapper,
                args=(i, task_params, request_queue, response_queue),
            )
            for i in range(n_process)
        ]
        for c in consumers:
            c.start()

        # Start output collector process (optional)
        collector = Process(
            target=postproc_func,
            args=(response_queue, output_filepath),
        )
        collector.start()

        # Start populating data
        for data in data_loader:
            request_queue.put(data)
        request_queue.put(None)

        for c in consumers:
            c.join()
        response_queue.put(None)

        collector.join()

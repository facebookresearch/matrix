# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import logging
import os
import socket
import time

import hydra
import numpy as np
import pandas as pd
import ray
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from ray.util.metrics import Counter, Gauge, Histogram

from matrix import Cli
from matrix.agents.examples.nr_curation import *
from matrix.client import query_llm
from matrix.utils.ray import get_ray_address, ray_get_async

logger = logging.getLogger(__name__)


# use Ray data, then async to allow control divergence
class NRInference:
    def __init__(
        self,
        cfg: DictConfig,
        filter_llm,
        score_llm,
        question_llm,
    ) -> None:
        self.cfg = cfg
        self.filter_llm = filter_llm
        self.score_llm = score_llm
        self.question_llm = question_llm

    def __call__(self, batch_df):
        """
        Args:
            batch_df: pandas DataFrame
        """

        async def _process_batch_async(rows):
            """Process all rows concurrently"""
            tasks = [self.process_single_row(row) for row in rows]
            results = await asyncio.gather(*tasks)
            return results

        # Convert to list of dicts
        rows = batch_df.to_dict("records")

        # Run async processing
        results = asyncio.run(_process_batch_async(rows))

        return pd.DataFrame(results)

    async def process_single_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single row through the multi-stage pipeline.

        Args:
            row: Dictionary representing a single record

        Returns:
            Dictionary with processing results for this row
        """

        def _post_process(response):
            response = response["response"]
            if "text" in response and isinstance(response["text"], list):
                response["text"] = response["text"][0]
            response["status_ok"] = "error" not in response
            return response

        def _make_requests(text: str, agent_cfg, llm_cfg):
            return {
                "data": {
                    "messages": [
                        {"role": "system", "content": agent_cfg.system_prompt},
                        {"role": "user", "content": text},
                    ]
                },
                **llm_cfg.sampling_params,
                **llm_cfg.exec_params,
            }

        def _clean_string(s: str) -> str:
            """Remove invalid Unicode characters including surrogates"""
            # Encode to UTF-8 with error handling, then decode back
            return s.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

        # Initialize result
        result = {
            "success": None,
            "termination_reason": None,
            "exam_question": "",
            "extracted_answer": "",
            "independent_answer": "",
        }

        # Stage 1: Filter by English language confidence
        lang_dict = row.get("language_id_whole_page_fasttext", {})
        if isinstance(lang_dict, dict):
            en_score = lang_dict.get("en", 0.0)
        else:
            en_score = 0.0

        if en_score < self.cfg.agents.filter.filter_en:
            result["success"] = False  # type: ignore[assignment]
            result["termination_reason"] = "filtered_by_language_check"
            return result

        text = row.get("text", "")

        # Stage 2: Filter stage (Yes/No filtering)
        response_stage1 = await query_llm.make_request(
            url=None,
            model=self.filter_llm["model_name"],
            **_make_requests(
                text, self.cfg.agents.filter, self.cfg.resources.filter_llm
            ),
            app_name=self.filter_llm["name"],
            endpoint_cache=self.filter_llm["endpoints"]["updater"],
        )
        response_stage1 = _post_process(response_stage1)
        processed_stage1 = FilterAgent.process_response(response_stage1)

        if not processed_stage1.get("success", True):
            result["success"] = False  # type: ignore[assignment]
            result["termination_reason"] = processed_stage1["termination_reason"]
            return result

        # Stage 3: Score inference call
        response_stage2 = await query_llm.make_request(
            url=None,
            model=self.score_llm["model_name"],
            **_make_requests(text, self.cfg.agents.score, self.cfg.resources.score_llm),
            app_name=self.score_llm["name"],
            endpoint_cache=self.score_llm["endpoints"]["updater"],
        )
        response_stage2 = _post_process(response_stage2)
        processed_stage2 = ScoreAgent.process_response(response_stage2, logger)

        if not processed_stage2.get("success", True):
            result["success"] = False  # type: ignore[assignment]
            result["termination_reason"] = processed_stage2["termination_reason"]
            return result

        # Stage 4: Question inference call
        response_stage3 = await query_llm.make_request(
            url=None,
            model=self.question_llm["model_name"],
            **_make_requests(
                text, self.cfg.agents.question, self.cfg.resources.question_llm
            ),
            app_name=self.question_llm["name"],
            endpoint_cache=self.question_llm["endpoints"]["updater"],
        )
        response_stage3 = _post_process(response_stage3)
        processed_stage3 = QuestionAgent.process_response(response_stage3, logger)

        result["success"] = processed_stage3.get("success", False)
        result["termination_reason"] = processed_stage3.get("termination_reason", "")
        if result["success"]:
            result["exam_question"] = _clean_string(
                processed_stage3.get("exam_question", "")
            )
            result["extracted_answer"] = _clean_string(
                processed_stage3.get("extracted_answer", "")
            )
            result["independent_answer"] = _clean_string(
                "\n".join(processed_stage3.get("independent_answer", []))
            )

        return result


@ray.remote
def run_remotely(cfg: DictConfig, filter_llm, score_llm, question_llm):
    logger.info(f"driver hostname is {socket.gethostname()}")
    is_zst = "zst" in cfg.dataset.data_files
    zst_files = glob.glob(os.path.expanduser(cfg.dataset.data_files))

    ds = ray.data.read_json(
        zst_files,
        arrow_open_stream_args={"compression": "zstd"} if is_zst else {},
        override_num_blocks=1000,  # Changed from parallelism=10
        ray_remote_args={"num_cpus": 1},  # Resources per read task
    )  # .limit(cfg.dataset.cut_off)

    response = ds.map_batches(
        NRInference,
        fn_constructor_kwargs={
            "cfg": cfg,
            "filter_llm": filter_llm,
            "score_llm": score_llm,
            "question_llm": question_llm,
        },
        batch_size=cfg.max_concurrent_tasks,
        num_gpus=0,
        num_cpus=1,
        concurrency=cfg.parallelism,
        batch_format="pandas",  # Get pandas DataFrame
    )
    # response.write_json(cfg.output.path, force_ascii=False)
    response.write_parquet(cfg.output.path, compression="zstd")


@hydra.main(config_path="../config", config_name="nr_curation", version_base=None)
def main(cfg: DictConfig):
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    cli = Cli(**cfg.matrix)
    if not ray.is_initialized():
        ray.init(
            address=get_ray_address(cli.cluster.cluster_info()),  # type: ignore[arg-type]
            log_to_driver=True,
        )
        if not cfg.debug:
            from ray.data import DataContext

            ctx = DataContext.get_current()
            ctx.enable_progress_bars = False

    filter_llm = cli.get_app_metadata(cfg.resources.filter_llm.matrix_service)
    score_llm = cli.get_app_metadata(cfg.resources.score_llm.matrix_service)
    question_llm = cli.get_app_metadata(cfg.resources.question_llm.matrix_service)

    start_time = time.time()
    ray.get(
        run_remotely.remote(
            cfg,
            filter_llm,
            score_llm,
            question_llm,
        )
    )
    print(f"Time taken: {time.time() - start_time} seconds")


if __name__ == "__main__":
    main()

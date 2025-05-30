# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time

# import argparse # Remove
import fire  # Add
import pandas as pd  # For add_column lambda
import ray
import ray.data

from .fit import generate_embeddings, generate_ump_embeddings

# Import shared utilities
from .utils import (
    CumlClusteringPredictor,
    UmapTransformer,
    get_outputs_path,
    inner_join,
    is_valid_parquet,
    is_valid_pickle,
    logger,
    summarize_clustering_stats,
)


@ray.remote  # type: ignore[arg-type]
def run_remotely(
    message_key: str,
    enable_umap: bool,
    cluster_alg: str,
    kmeans_path: str,
    kmeans_model_path: str,
    hdbscan_path: str,
    viz_path: str,
    umap_cluster_model_path: str,
    umap_viz_model_path: str,
    hdbscan_model_path: str,
    embeddings_path: str,
    umap_embeddings_path: str,
    umap_viz_embeddings_path: str,
    uuid_ds_path: str,
    input_jsonl: str,
    text_key: str,
    embedding_model: str,
    max_concurrency: int,
    batch_size: int,
    viz_sample_size: int,
):
    uuid_ds, embedded_ds = generate_embeddings(
        input_jsonl,
        uuid_ds_path,
        embeddings_path,
        max_concurrency,
        embedding_model,
        text_key,
        message_key,
        batch_size,
    )
    if enable_umap:
        cluster_embedding_ds = generate_ump_embeddings(
            embedded_ds,
            umap_cluster_model_path,
            umap_embeddings_path,
            max_concurrency,
            batch_size,
        ).rename_columns({"umap_embedding": "embedding"})
    else:
        cluster_embedding_ds = embedded_ds

    umap_viz_embedding_ds = generate_ump_embeddings(
        embedded_ds,
        umap_viz_model_path,
        umap_viz_embeddings_path,
        max_concurrency,
        batch_size,
    )

    # --- 3. Predict using HDBSCAN ---
    if cluster_alg == "hdbscan":
        if not is_valid_parquet(hdbscan_path):
            logger.info("Starting HDBSCAN prediction...")
            start_time = time.time()

            results_ds = cluster_embedding_ds.map_batches(
                CumlClusteringPredictor,
                fn_constructor_kwargs={
                    "model_path": hdbscan_model_path,
                    "model_type": "HDBSCAN",
                },
                batch_size=batch_size,
                concurrency=max_concurrency,
                num_gpus=1,
                batch_format="numpy",
                # Input: {'id': ..., 'umap_embedding': ...}
                # Output: {'id': ..., 'cluster_label': ...}
            )
            logger.info(
                f"HDBSCAN prediction finished in {time.time() - start_time:.2f}s."
            )

            logger.info(f"Saving inference results to {hdbscan_path}...")
            start_time = time.time()
            # Results ds should have 'id' and 'cluster_label'
            results_ds.write_parquet(hdbscan_path)
            logger.info(f"Results saved in {time.time() - start_time:.2f}s.")
        labels_ds = ray.data.read_parquet(hdbscan_path)
    elif cluster_alg == "kmeans":
        if not is_valid_parquet(kmeans_path):
            logger.info("Starting kmeans prediction...")
            start_time = time.time()

            results_ds = cluster_embedding_ds.map_batches(
                CumlClusteringPredictor,
                fn_constructor_kwargs={
                    "model_path": kmeans_model_path,
                    "model_type": "KMeans",
                },
                batch_size=batch_size,
                concurrency=max_concurrency,
                num_gpus=1,
                batch_format="numpy",
                # Input: {'id': ..., 'umap_embedding': ...}
                # Output: {'id': ..., 'cluster_label': ...}
            )
            logger.info(
                f"Kmeans prediction finished in {time.time() - start_time:.2f}s."
            )

            logger.info(f"Saving inference results to {kmeans_path}...")
            start_time = time.time()
            # Results ds should have 'id' and 'cluster_label'
            results_ds.write_parquet(kmeans_path)
            logger.info(f"Results saved in {time.time() - start_time:.2f}s.")
        labels_ds = ray.data.read_parquet(kmeans_path)
    else:
        assert False

    summarize_clustering_stats(labels_ds)

    # --- 4. Join Labels and 2D Coords ---
    if viz_sample_size > 0 and not is_valid_pickle(viz_path):
        # ray does not support join yet https://github.com/ray-project/ray/issues/18911
        logger.info("Joining labels and visualization coordinates...")
        start_time = time.time()
        labels_ds = labels_ds.filter(lambda row: row["cluster_label"] != -1)
        fraction = min(viz_sample_size / labels_ds.count(), 1.0)
        sampled_ds = labels_ds.random_sample(fraction=fraction)

        # Select columns before join for efficiency
        viz_coords_to_join = umap_viz_embedding_ds.select_columns(
            ["id", "umap_embedding"]
        ).rename_columns({"umap_embedding": "umap_viz"})
        final_viz_ds = inner_join(
            sampled_ds,
            viz_coords_to_join,
            key="id",
        )
        print(f"joined {final_viz_ds.count()} rows")
        final_viz_ds = inner_join(final_viz_ds, uuid_ds, key="id")
        print(f"joined {final_viz_ds.count()} rows")
        final_viz_df = final_viz_ds.to_pandas()
        final_viz_df = final_viz_df.sort_values(by="cluster_label")
        final_viz_df.to_pickle(viz_path)
        logger.info(f"Results saved in {time.time() - start_time:.2f}s.")


def main(
    # Required arguments
    ray_head_url,
    input_jsonl: str,
    artifact_dir: str,
    # Optional arguments
    run_id: str = "0",
    text_key: str = "src",
    message_key: str = "",
    embedding_model: str = "all-MiniLM-L6-v2",  # Needed if input_type is 'text'
    enable_umap: bool = False,
    umap_cluster_dim: int = 30,
    cluster_alg: str = "kmeans",
    kmeans_num_clusters: int = 1000,
    hdbscan_min_cluster_size: int = 100,
    hdbscan_min_samples: int = 10,
    max_concurrency: int = 8,
    batch_size: int = 1024,
    umap_viz_dim: int = 2,
    viz_sample_size: int = 0,
):
    """
    Apply fitted UMAP and HDBSCAN models for text clustering inference.

    Args:
        input_path: Path pattern to input text files or saved embeddings parquet. Required.
        artifact_dir: Directory containing saved models from the fit stage. Required.
        run_id: Run ID used during the fitting stage to identify models. Required.
        output_path: Path to save inference results (parquet format). Required.
        input_type: 'text' or 'embeddings'. If 'embeddings', input_path must point to parquet file.
        input_format: Format of input files ('text', 'json', 'parquet'), used if input_type='text'.
        text_column: Column name containing text (if text input and json/parquet format).
        embedding_model: Sentence Transformer model name (Needed if input_type is 'text').
        num_gpus_per_worker: Number of GPUs per Ray worker for parallel tasks.
        num_actors: Number of concurrent actors for map_batches.
        batch_size: Batch size for embedding (if text input).
    """
    assert ":" in ray_head_url, "ray_head_url should be in the format of hostname:port"
    if not ray_head_url.startswith("ray://"):
        ray_head_url = f"ray://{ray_head_url}"

    logger.info(f"Starting Inference Pipeline - Run ID: {run_id}")
    logger.info(
        f"Parameters: input_path='{input_jsonl}', "
        f"artifact_dir='{artifact_dir}', ..."
    )  # Log key params

    os.makedirs(artifact_dir, exist_ok=True)

    outputs_path = get_outputs_path(
        artifact_dir,
        run_id,
        embedding_model,
        enable_umap,
        cluster_alg,
        umap_cluster_dim,
        umap_viz_dim,
        sample_size=None,
        params={
            "kmeans_num_clusters": kmeans_num_clusters,
            "hdbscan_min_cluster_size": hdbscan_min_cluster_size,
            "hdbscan_min_samples": hdbscan_min_samples,
        },
    )
    uuid_ds_path = outputs_path["uuid_ds_path"]
    embeddings_path = outputs_path["embeddings_path"]
    umap_cluster_model_path = outputs_path["umap_cluster_model_path"]
    umap_viz_model_path = outputs_path["umap_viz_model_path"]
    umap_embeddings_path = outputs_path["umap_embeddings_path"]
    umap_viz_embeddings_path = outputs_path["umap_viz_embeddings_path"]
    hdbscan_model_path = outputs_path["hdbscan_model_path"]
    kmeans_model_path = outputs_path["kmeans_model_path"]
    hdbscan_path = outputs_path["hdbscan_path"]
    kmeans_path = outputs_path["kmeans_path"]
    viz_path = outputs_path["viz_path"]

    if enable_umap and not os.path.exists(umap_cluster_model_path):
        raise FileNotFoundError(f"No file found {umap_cluster_model_path}")
    if not os.path.exists(umap_viz_model_path):
        raise FileNotFoundError(f"No file found {umap_viz_model_path}")
    if cluster_alg == "hdbscan" and not os.path.exists(hdbscan_model_path):
        raise FileNotFoundError(f"No file found {hdbscan_model_path}")
    elif cluster_alg == "kmeans" and not os.path.exists(kmeans_model_path):
        raise FileNotFoundError(f"No file found {kmeans_model_path}")

    if not ray.is_initialized():
        ray.init(address=ray_head_url, log_to_driver=True)
    logger.info(f"Ray Initialized: {ray.cluster_resources()}")

    print("Launching remote task...")
    future_result = run_remotely.remote(  # type: ignore[call-arg]
        message_key,
        enable_umap,
        cluster_alg,
        kmeans_path,
        kmeans_model_path,
        hdbscan_path,
        viz_path,
        umap_cluster_model_path,
        umap_viz_model_path,
        hdbscan_model_path,
        embeddings_path,
        umap_embeddings_path,
        umap_viz_embeddings_path,
        uuid_ds_path,
        input_jsonl,
        text_key,
        embedding_model,
        max_concurrency,
        batch_size,
        viz_sample_size,
    )

    # --- Get the result and observe logs ---
    # While this line is waiting, you should see the print and logging
    # output from the remote task appear in your local console.
    print("Waiting for remote task to complete...")
    result = ray.get(future_result)

    logger.info(f"Inference Pipeline Completed Successfully for Run ID: {run_id}")


if __name__ == "__main__":
    fire.Fire(main)

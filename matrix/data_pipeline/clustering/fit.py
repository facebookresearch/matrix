# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import uuid

# import argparse # Remove
import fire  # Add
import numpy as np
import pandas as pd  # Needed for add_column lambda
import ray
import ray.data

# Import shared utilities
from .utils import (
    LLMEmbedder,
    SentenceEmbedder,
    UmapTransformer,
    fit_hdbscan_on_worker_gpu,
    fit_kmeans_on_worker_gpu,
    fit_umap_on_worker_gpu,
    get_nested_value,
    get_outputs_path,
    is_valid_parquet,
    is_valid_pickle,
    logger,
    save_model,
    xxhash128,
)


def generate_embeddings(
    input_jsonl,
    uuid_ds_path,
    embeddings_path,
    max_concurrency,
    embedding_model,
    text_key,
    message_key,
    batch_size_embed,
):
    # --- 1. Load Data ---
    if not is_valid_parquet(uuid_ds_path):
        logger.info(f"Loading data from: {input_jsonl}")
        ds = ray.data.read_json(input_jsonl)
        logger.info(ds.schema())

        def add_uuid8_column(batch: pd.DataFrame) -> pd.DataFrame:
            batch = batch.copy()
            texts = batch[text_key].tolist()  # Extract column values as a list
            batch["_text"] = texts
            batch["id"] = [xxhash128(text) for text in texts]
            return batch

        def extract_batch(batch: pd.DataFrame) -> pd.DataFrame:
            """Extract and hash message content using message_key; keep all other columns. Add '_text' and 'id'."""
            batch = batch.copy()

            def extract_and_hash(row):
                text = get_nested_value(row.to_dict(), message_key)
                return pd.Series({"_text": text, "id": xxhash128(text)})

            new_cols = batch.apply(extract_and_hash, axis=1)
            batch[["_text", "id"]] = new_cols
            return batch

        if text_key:
            ds = ds.map_batches(add_uuid8_column, batch_format="pandas")
        elif message_key:
            ds = ds.map_batches(extract_batch, batch_format="pandas")
        else:
            assert False

        ds.write_parquet(uuid_ds_path)
    ds = ray.data.read_parquet(uuid_ds_path)

    # --- 2. Embedding ---
    if not is_valid_parquet(embeddings_path):
        logger.info(f"Starting embedding {embeddings_path}... ")
        start_time = time.time()
        is_llm = "/" in embedding_model.lower()
        embedded_ds = ds.select_columns(["id", "_text"]).map_batches(
            LLMEmbedder if is_llm else SentenceEmbedder,
            fn_constructor_kwargs={
                "model_name": embedding_model,
                "text_key": "_text",
            },
            batch_size=min(64, batch_size_embed) if is_llm else batch_size_embed,
            concurrency=max_concurrency,
            num_gpus=1,  # embedding model is very small
            batch_format="numpy",
            # fn_kwargs={}, # Not needed here
        )
        logger.info(f"Embedding finished in {time.time() - start_time:.2f}s.")

        logger.info(f"Saving embeddings to {embeddings_path}...")
        start_time = time.time()
        # Select only relevant columns for saving
        embedded_ds.write_parquet(embeddings_path)
        logger.info(f"Embeddings saved in {time.time() - start_time:.2f}s.")

    embedded_ds = ray.data.read_parquet(embeddings_path)
    return ds, embedded_ds


def generate_ump_embeddings(
    embedded_ds,
    umap_cluster_model_path,
    umap_embeddings_path,
    max_concurrency,
    batch_size_umap_transform,
):
    if not is_valid_parquet(umap_embeddings_path):
        logger.info(f"Transforming UMAP sample for HDBSCAN fitting...")
        start_time = time.time()

        # We need the *original* sample embeddings + id to transform
        # Re-use umap_fit_ds_ref definition but include id earlier, or re-sample embedded_ds
        umap_sample_with_id_ref = (
            embedded_ds.select_columns(["id", "embedding"])
            # .random_shuffle()
            # .limit(umap_sample_size)
        )

        umap_embedding_ds = umap_sample_with_id_ref.map_batches(
            UmapTransformer,
            fn_constructor_kwargs={"model_path": umap_cluster_model_path},
            batch_size=batch_size_umap_transform,
            concurrency=max_concurrency,
            num_gpus=1,
            batch_format="numpy",
        )
        logger.info(
            f"UMAP sample transformation finished in {time.time() - start_time:.2f}s."
        )
        logger.info(f"Saving embeddings to {umap_embeddings_path}...")
        start_time = time.time()
        # Select only relevant columns for saving
        umap_embedding_ds.write_parquet(umap_embeddings_path)
        logger.info(f"Embeddings saved in {time.time() - start_time:.2f}s.")
    umap_embedding_ds = ray.data.read_parquet(umap_embeddings_path)
    return umap_embedding_ds


@ray.remote  # type: ignore[arg-type]
def run_remotely(
    umap_cluster_model_path: str,
    umap_viz_model_path: str,
    hdbscan_model_path: str,
    embeddings_path: str,
    umap_embeddings_path: str,
    uuid_ds_path: str,
    kmeans_model_path,
    enable_umap,
    cluster_alg,
    kmeans_num_clusters,
    input_jsonl: str,
    text_key: str,
    message_key: str,
    embedding_model: str,
    umap_metric: str,
    umap_cluster_dim: int,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
    umap_fit_sample_frac: float,
    clustering_fit_sample_frac: float,
    max_concurrency: int,
    batch_size: int,
    umap_viz_dim: int,
    run_id: str,
):

    ds, embedded_ds = generate_embeddings(
        input_jsonl,
        uuid_ds_path,
        embeddings_path,
        max_concurrency,
        embedding_model,
        text_key,
        message_key,
        batch_size,
    )
    total_rows = embedded_ds.count()

    # --- 3. Sample for UMAP Fitting ---
    if (
        enable_umap and not is_valid_pickle(umap_cluster_model_path)
    ) or not is_valid_pickle(umap_viz_model_path):
        if total_rows == 0:
            logger.error("Dataset is empty after embedding. Exiting.")
            return

        umap_sample_size = int(total_rows * umap_fit_sample_frac)
        if umap_sample_size == 0 and total_rows > 0:
            umap_sample_size = 1
        logger.info(
            f"Sampling {umap_sample_size} points ({umap_fit_sample_frac*100:.1f}%) for UMAP fitting..."
        )
        umap_fit_ds_ref = (
            embedded_ds.select_columns(["embedding"])
            .random_shuffle()
            .limit(umap_sample_size)
        ).materialize()  # Select only embedding

        # --- 4. Fit UMAP Models ---
        logger.info("Scheduling UMAP fitting tasks...")
        umap_tasks = []
        if enable_umap and not is_valid_pickle(umap_cluster_model_path):
            umap_cluster_task = fit_umap_on_worker_gpu.remote(  # type: ignore[call-arg]
                umap_fit_ds_ref,
                umap_cluster_dim,
                umap_cluster_model_path,
                f"cluster_{run_id}",
                umap_metric,
            )
            umap_tasks.append(umap_cluster_task)
        if not is_valid_pickle(umap_viz_model_path):
            umap_viz_task = fit_umap_on_worker_gpu.remote(  # type: ignore[call-arg]
                umap_fit_ds_ref,
                umap_viz_dim,
                umap_viz_model_path,
                f"viz_{run_id}",
                umap_metric,
            )
            umap_tasks.append(umap_viz_task)
        start_time = time.time()
        ray.get(umap_tasks)
        logger.info(
            f"UMAP model fitting tasks completed and models saved in {time.time() - start_time:.2f}s."
        )

    # --- 5. Transform UMAP Sample to Clustering Dimensions ---
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

    # --- 6. Sample Low-Dim Data for HDBSCAN Fitting ---
    if cluster_alg == "hdbscan" and not is_valid_pickle(hdbscan_model_path):
        hdbscan_fit_input_count = cluster_embedding_ds.count()
        hdbscan_sample_size = int(total_rows * clustering_fit_sample_frac)
        if hdbscan_sample_size == 0 and hdbscan_fit_input_count > 0:
            hdbscan_sample_size = 1
        logger.info(
            f"Sampling {hdbscan_sample_size} points ({clustering_fit_sample_frac*100:.1f}%) of UMAP sample for HDBSCAN fitting..."
        )
        # Select only the umap_embedding column needed for fitting
        hdbscan_fit_ds_ref = (
            cluster_embedding_ds.select_columns(["embedding"])
            .random_shuffle()
            .limit(hdbscan_sample_size)
        ).materialize()

        # --- 7. Fit HDBSCAN ---
        logger.info("Scheduling HDBSCAN fitting task...")
        hdbscan_task = fit_hdbscan_on_worker_gpu.remote(  # type: ignore[call-arg]
            hdbscan_fit_ds_ref,
            hdbscan_model_path,
            hdbscan_min_cluster_size,
            hdbscan_min_samples,
        )

        start_time = time.time()
        ray.get(hdbscan_task)
        logger.info(
            f"HDBSCAN fitting task completed and model saved in {time.time() - start_time:.2f}s."
        )

    if cluster_alg == "kmeans" and not is_valid_pickle(kmeans_model_path):
        kmeans_fit_input_count = cluster_embedding_ds.count()
        kmeans_sample_size = int(total_rows * clustering_fit_sample_frac)
        if kmeans_sample_size == 0 and kmeans_fit_input_count > 0:
            kmeans_sample_size = 1
        logger.info(
            f"Sampling {kmeans_sample_size} points ({clustering_fit_sample_frac*100:.1f}%) of UMAP sample for kmeans fitting..."
        )
        # Select only the umap_embedding column needed for fitting
        kmeans_fit_ds_ref = (
            cluster_embedding_ds.select_columns(["embedding"])
            .random_shuffle()
            .limit(kmeans_sample_size)
        ).materialize()

        # --- 7. Fit kmeans ---
        logger.info("Scheduling kmeans fitting task...")
        kmeans_task = fit_kmeans_on_worker_gpu.remote(  # type: ignore[call-arg]
            kmeans_fit_ds_ref,
            kmeans_model_path,
            kmeans_num_clusters,
        )

        start_time = time.time()
        ray.get(kmeans_task)
        logger.info(
            f"kmeans fitting task completed and model saved in {time.time() - start_time:.2f}s."
        )


def main(
    ray_head_url,
    input_jsonl,
    artifact_dir: str,
    text_key: str = "src",
    message_key: str = "",
    # embedding
    embedding_model: str = "all-MiniLM-L6-v2",  # all-mpnet-base-v2
    # umap
    enable_umap: bool = False,
    umap_metric="euclidean",
    umap_cluster_dim: int = 30,
    # clustering
    cluster_alg: str = "kmeans",
    kmeans_num_clusters: int = 1000,
    hdbscan_min_cluster_size: int = 100,
    hdbscan_min_samples: int = 10,
    # default
    umap_fit_sample_frac: float = 1.0,
    clustering_fit_sample_frac: float = 1.0,
    max_concurrency: int = 8,
    batch_size: int = 1024,
    umap_viz_dim: int = 2,
    run_id: str = "0",
):
    """
    Fit UMAP and HDBSCAN models on text data using Ray.

    Args:
        ray_head_url: Ray head (hostname:client_server_port), eg localhost:10001
        input_jsonl: Path pattern to input text files (e.g., 's3://bucket/data/*.txt'). Required.
        text_key: Column name containing text (if input is json/parquet).
        artifact_dir: Directory to save fitted models and intermediate data.
        embedding_model: Sentence Transformer model name.
        umap_neighbors: UMAP n_neighbors.
        umap_min_dist: UMAP min_dist.
        umap_cluster_dim: UMAP target dimensions for clustering.
        hdbscan_min_cluster_size: HDBSCAN min_cluster_size.
        hdbscan_min_samples: HDBSCAN min_samples.
        umap_fit_sample_frac: Fraction of data to sample for UMAP fitting (0.0 to 1.0).
        clustering_fit_sample_frac: Fraction of UMAP-transformed data for HDBSCAN fitting (0.0 to 1.0).
        num_gpus_per_worker: Number of GPUs per Ray worker for parallel tasks.
        max_concurrency: Number of concurrent embedding actors.
        batch_size: Batch size for embedding generation.
        save_embeddings: If True, save generated embeddings to artifact directory.
        umap_viz_dim: UMAP target dimensions for visualization.
    """
    assert ":" in ray_head_url, "ray_head_url should be in the format of hostname:port"
    if not ray_head_url.startswith("ray://"):
        ray_head_url = f"ray://{ray_head_url}"

    # run_id = str(uuid.uuid4())[:8]
    logger.info(f"Starting Fit Pipeline - Run ID: {run_id}")
    # Log parameters directly
    logger.info(
        f"Parameters: input_path='{input_jsonl}', artifact_dir='{artifact_dir}', "
        f"embedding_model='{embedding_model}', umap_cluster_dim={umap_cluster_dim}"
    )  # Log key params

    # Ensure artifact directory exists
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
    hdbscan_model_path = outputs_path["hdbscan_model_path"]
    kmeans_model_path = outputs_path["kmeans_model_path"]

    # --- Ray Init ---
    if not ray.is_initialized():
        ray.init(address=ray_head_url, log_to_driver=True)
    logger.info(f"Ray Initialized: {ray.cluster_resources()}")

    print("Launching remote task...")
    future_result = run_remotely.remote(  # type: ignore[call-arg]
        umap_cluster_model_path,
        umap_viz_model_path,
        hdbscan_model_path,
        embeddings_path,
        umap_embeddings_path,
        uuid_ds_path,
        kmeans_model_path,
        enable_umap,
        cluster_alg,
        kmeans_num_clusters,
        input_jsonl,
        text_key,
        message_key,
        embedding_model,
        umap_metric,
        umap_cluster_dim,
        hdbscan_min_cluster_size,
        hdbscan_min_samples,
        umap_fit_sample_frac,
        clustering_fit_sample_frac,
        max_concurrency,
        batch_size,
        umap_viz_dim,
        run_id,
    )

    # --- Get the result and observe logs ---
    # While this line is waiting, you should see the print and logging
    # output from the remote task appear in your local console.
    print("Waiting for remote task to complete...")
    result = ray.get(future_result)

    print(f"Local script received result: {result}")

    logger.info(f"Fit Pipeline Completed Successfully for Run ID: {run_id}")
    logger.info(f"Fitted models saved in: {artifact_dir}")
    # Print model paths for user convenience
    logger.info(
        f"  UMAP Cluster Model ({umap_cluster_dim}D): {umap_cluster_model_path}"
    )
    logger.info(f"  UMAP Viz Model ({umap_viz_dim}D): {umap_viz_model_path}")
    logger.info(f"  HDBSCAN Model: {hdbscan_model_path}")
    logger.info(f"  Embeddings: {embeddings_path}")
    logger.info(f"Use Run ID '{run_id}' for inference and visualization.")


if __name__ == "__main__":
    # fire.Fire(main) expects main to be defined
    fire.Fire(main)

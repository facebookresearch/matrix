# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import concurrent.futures
import math
import os
import random
import time
from typing import Dict, List, Tuple

# import argparse # Remove
import fire  # Add
import numpy as np
import pandas as pd  # For add_column lambda
import ray
import ray.data
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import pairwise_distances

from .fit import generate_embeddings, generate_ump_embeddings

# Import shared utilities
from .utils import get_outputs_path, inner_join, logger, summarize_clustering_stats


@ray.remote
def run_remotely(
    output_jsonl: str,
    cluster_path: str,
    embeddings_path: str,
    uuid_ds_path: str,
    sample_size: int,
    same_samples_per_cluster: bool,
    random_sample_within_cluster: bool,
    seed: int = 42,
):
    uuid_ds = ray.data.read_parquet(uuid_ds_path)
    embedding_ds = ray.data.read_parquet(embeddings_path)
    if "umap_embedding" in embedding_ds.schema().names:
        embedding_ds = embedding_ds.rename_columns({"umap_embedding": "embedding"})
    labels_ds = ray.data.read_parquet(cluster_path)
    labels_ds = labels_ds.filter(lambda row: row["cluster_label"] != -1)

    summarize_clustering_stats(labels_ds)

    if random_sample_within_cluster:
        label_embedding_ds = labels_ds
    else:
        label_embedding_ds = inner_join(
            labels_ds,
            embedding_ds,
            key="id",
        )

    np.random.seed(seed)
    random.seed(seed)
    # Step 1: Group by cluster_label to get cluster sizes
    cluster_counts = label_embedding_ds.groupby("cluster_label").count()
    cluster_sizes = {
        row["cluster_label"]: row["count()"] for row in cluster_counts.take_all()
    }
    num_clusters = len(cluster_sizes)

    # Step 2: Determine sample size per cluster
    samples_per_cluster = {}
    remaining_samples = sample_size

    # First, ensure we have at least one sample from each cluster if possible
    if same_samples_per_cluster or sample_size <= num_clusters:
        samples = np.random.choice(
            list(cluster_sizes.keys()), size=sample_size, replace=True
        )
        # Count occurrences of each item
        samples_per_cluster = dict(collections.Counter(samples))
    else:
        # At least one sample from each cluster
        samples_per_cluster = {cluster_id: 1 for cluster_id in cluster_sizes.keys()}
        remaining_samples -= num_clusters

        # For remaining samples, allocate based on sqrt(cluster_size)
        total_sqrt_size = sum(math.sqrt(size) for size in cluster_sizes.values())

        # Calculate initial allocation (float values)
        float_allocations = {
            cluster_id: math.sqrt(size) / total_sqrt_size * remaining_samples
            for cluster_id, size in cluster_sizes.items()
        }

        # Convert to integers while preserving total sample size
        int_allocations = {
            cluster_id: int(allocation)
            for cluster_id, allocation in float_allocations.items()
        }
        remaining_after_int = remaining_samples - sum(int_allocations.values())

        # Distribute remaining samples based on fractional parts
        fractional_parts = {
            cluster_id: allocation - int(allocation)
            for cluster_id, allocation in float_allocations.items()
        }

        for cluster_id, _ in sorted(
            fractional_parts.items(), key=lambda x: x[1], reverse=True
        )[:remaining_after_int]:
            int_allocations[cluster_id] += 1

        # Add to the base allocation of 1 per cluster
        for cluster_id, additional in int_allocations.items():
            samples_per_cluster[cluster_id] += additional

    # Step 3: For each cluster, select diverse samples
    # Define a function to apply to each cluster group in parallel
    def select_sample_ids_from_cluster(cluster_df):
        cluster_id = cluster_df["cluster_label"][0]
        if (
            cluster_id not in samples_per_cluster
            or samples_per_cluster[cluster_id] == 0
        ):
            return []

        k = samples_per_cluster[cluster_id]
        k = min(k, len(cluster_df["id"]))

        ids = np.array(cluster_df["id"])
        if random_sample_within_cluster:
            selected_ids = np.random.choice(ids, size=k, replace=False)
        else:
            embeddings = np.vstack(cluster_df["embedding"])
            centroid = np.mean(embeddings, axis=0)
            logger.info(embeddings.shape, centroid.shape)
            distances_to_centroid = np.linalg.norm(embeddings - centroid, axis=1)

            # First pick the closest point to centroid
            selected_indices = [np.argmin(distances_to_centroid)]
            selected_ids = [ids[selected_indices[0]]]

            if k > 1:
                distance_matrix = squareform(pdist(embeddings))

                while len(selected_indices) < k:
                    if len(selected_indices) == 1:
                        min_distances = distance_matrix[selected_indices[0]]
                    else:
                        min_distances = np.min(
                            [distance_matrix[idx] for idx in selected_indices], axis=0
                        )

                    for idx in selected_indices:
                        min_distances[idx] = -1

                    # Select the point with maximum minimum distance
                    next_idx = np.argmax(min_distances)
                    selected_indices.append(next_idx)
                    selected_ids.append(ids[next_idx])

        return pd.DataFrame(selected_ids, columns=["selected_id"])

    # Use map_groups to parallelize the sampling process across clusters
    sampled_ids_ds = label_embedding_ds.groupby("cluster_label").map_groups(
        select_sample_ids_from_cluster,
    )

    # Extract the sampled IDs
    sampled_ids = set([row["selected_id"] for row in sampled_ids_ds.take_all()])

    # Filter the original dataset to get only the sampled IDs
    sampled_ds = uuid_ds.filter(lambda row: row["id"] in sampled_ids)

    # Write results to output
    sampled_ds.to_pandas().to_json(output_jsonl, orient="records", lines=True)
    return (
        f"Successfully sampled {len(sampled_ids)} items across {num_clusters} clusters"
    )


def main(
    # Required arguments
    ray_head_url,
    inference_dir: str,
    # Optional arguments
    embedding_model: str = "all-MiniLM-L6-v2",  # all-mpnet-base-v2
    enable_umap: bool = False,
    umap_cluster_dim: int = 30,
    umap_viz_dim: int = 2,
    cluster_alg: str = "kmeans",
    kmeans_num_clusters: int = 1000,
    hdbscan_min_cluster_size: int = 100,
    hdbscan_min_samples: int = 10,
    run_id: str = "0",
    sample_size: int = 1000,
    same_samples_per_cluster: bool = True,
    random_sample_within_cluster: bool = True,
    seed: int = 42,
):
    """
    Sample from the clustered results.

    Args:
        run_id: Run ID used during the fitting stage to identify models. Required.
        output_dir: Path to save inference results (parquet format). Required.
        sample_size: Num of samples desired.
    """
    assert ":" in ray_head_url, "ray_head_url should be in the format of hostname:port"
    if not ray_head_url.startswith("ray://"):
        ray_head_url = f"ray://{ray_head_url}"

    logger.info(f"Starting Sample Pipeline - Run ID: {run_id}")
    logger.info(f"Parameters: inference_dir='{inference_dir}'")  # Log key params

    output_dir = inference_dir
    os.makedirs(output_dir, exist_ok=True)
    outputs_path = get_outputs_path(
        inference_dir,
        run_id,
        embedding_model,
        enable_umap,
        cluster_alg,
        umap_cluster_dim,
        umap_viz_dim,
        sample_size,
        params={
            "kmeans_num_clusters": kmeans_num_clusters,
            "hdbscan_min_cluster_size": hdbscan_min_cluster_size,
            "hdbscan_min_samples": hdbscan_min_samples,
        },
    )
    uuid_ds_path = outputs_path["uuid_ds_path"]
    embeddings_path = outputs_path["embeddings_path"]
    umap_embeddings_path = outputs_path["umap_embeddings_path"]
    hdbscan_path = outputs_path["hdbscan_path"]
    kmeans_path = outputs_path["kmeans_path"]
    sample_jsonl = outputs_path["sample_jsonl"]
    name, ext = os.path.splitext(sample_jsonl)
    sample_jsonl = (
        f"{name}_{same_samples_per_cluster}_{random_sample_within_cluster}{ext}"
    )

    if not ray.is_initialized():
        ray.init(address=ray_head_url, log_to_driver=True)
    logger.info(f"Ray Initialized: {ray.cluster_resources()}")

    print("Launching remote task...")
    future_result = run_remotely.remote(
        sample_jsonl,
        kmeans_path if cluster_alg == "kmeans" else hdbscan_path,
        umap_embeddings_path if enable_umap else embeddings_path,
        uuid_ds_path,
        sample_size,
        same_samples_per_cluster,
        random_sample_within_cluster,
        seed,
    )

    print("Waiting for remote task to complete...")
    result = ray.get(future_result)

    logger.info(f"Sampling Pipeline Completed Successfully for Run ID: {run_id}")


if __name__ == "__main__":
    fire.Fire(main)

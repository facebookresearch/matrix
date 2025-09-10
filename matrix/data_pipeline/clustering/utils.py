# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import logging
import os
import pickle
import re
import typing as tp

import cloudpickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import ray
import torch
from sentence_transformers import SentenceTransformer

from matrix.utils.basics import get_nested_value, get_user_message_from_llama3_prompt

# Basic Logging Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Model Saving/Loading ---


import xxhash


def get_outputs_path(
    artifact_dir,
    run_id,
    embedding_model: str,
    enable_umap: bool,
    cluster_alg,
    umap_cluster_dim=30,
    umap_viz_dim=2,
    sample_size=1000,
    params={},
):
    embedding_model = embedding_model.replace("/", "_")
    umap_name = "umap" if enable_umap else "no_umap"
    if cluster_alg == "kmeans":
        cluster_params = params["kmeans_num_clusters"]
    else:
        min_cluster = params["hdbscan_min_cluster_size"]
        min_sample = params["hdbscan_min_samples"]
        cluster_params = f"min_cluster_{min_cluster}_min_sample_{min_sample}"
    uuid_ds_path = os.path.join(artifact_dir, f"uuid_{run_id}.parquet")
    embeddings_path = os.path.join(
        artifact_dir, f"embeddings_{run_id}_{embedding_model}.parquet"
    )
    umap_cluster_model_path = os.path.join(
        artifact_dir, f"umap_model_{umap_cluster_dim}D_{run_id}_{embedding_model}.pkl"
    )
    umap_viz_model_path = os.path.join(
        artifact_dir, f"umap_model_{umap_viz_dim}D_{run_id}_{embedding_model}.pkl"
    )
    umap_embeddings_path = os.path.join(
        artifact_dir,
        f"umap_model_{umap_cluster_dim}D_{run_id}_{embedding_model}.parquet",
    )
    umap_viz_embeddings_path = os.path.join(
        artifact_dir, f"umap_model_{umap_viz_dim}D_{run_id}_{embedding_model}.parquet"
    )
    hdbscan_model_path = os.path.join(
        artifact_dir, f"hdbscan_model_{run_id}_{embedding_model}_{umap_name}.pkl"
    )
    kmeans_model_path = os.path.join(
        artifact_dir,
        f"kmeans_model_{run_id}_{cluster_params}_{embedding_model}_{umap_name}.pkl",
    )
    hdbscan_path = os.path.join(
        artifact_dir, f"hdbscan_{run_id}_{embedding_model}_{umap_name}.pkl"
    )
    kmeans_path = os.path.join(
        artifact_dir,
        f"kmeans_{run_id}_{cluster_params}_{embedding_model}_{umap_name}",
    )
    viz_path = os.path.join(
        artifact_dir,
        f"viz_{run_id}_{embedding_model}_{umap_name}_{cluster_alg}_{cluster_params}.pkl",
    )
    sample_jsonl = os.path.join(
        artifact_dir,
        f"sampled_{run_id}_{sample_size}_{embedding_model}_{umap_name}_{cluster_alg}_{cluster_params}.jsonl",
    )

    return locals()


def xxhash128(text: str) -> str:
    return xxhash.xxh3_128_hexdigest(text)


def is_valid_parquet(path: str) -> bool:
    if not os.path.exists(path):
        return False
    if not os.path.isdir(path):
        return False
    try:
        # Check if directory has at least one valid parquet file
        for file in os.listdir(path):
            if file.endswith(".parquet"):
                pq.read_table(os.path.join(path, file), columns=[])
                return True
    except Exception as e:
        logger.warning(f"Found embeddings path but failed to read: {e}")
        return False
    return False


def is_valid_pickle(path: str) -> bool:
    """
    Checks if a file contains valid pickle data that can be successfully loaded.
    """
    if not os.path.exists(path):
        return False

    try:
        with open(path, "rb") as f:
            return True

    except (pickle.UnpicklingError, EOFError, OSError) as e:
        return False
    except Exception as e:
        return False


def save_model(model, path):
    """Saves a model using cloudpickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "wb") as f:
            cloudpickle.dump(model, f)
        logger.info(f"Model successfully saved to {path}")
    except Exception as e:
        logger.error(f"Error saving model to {path}: {e}", exc_info=True)
        raise


def load_model(path):
    """Loads a model using cloudpickle."""
    try:
        with open(path, "rb") as f:
            model = cloudpickle.load(f)
        logger.info(f"Model successfully loaded from {path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model from {path}: {e}", exc_info=True)
        raise


# --- Ray Data Actor Classes ---


class SentenceEmbedder:
    def __init__(self, model_name, text_key):
        # Loads model onto the GPU assigned to this actor by Ray
        self.model = SentenceTransformer(model_name, device="cuda")
        self.text_key = text_key
        logger.info(f"Embedder actor initialized with model {model_name} on GPU.")

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # Assumes 'id' and 'text' columns in the input batch (numpy format)
        ids = batch["id"]
        texts = [
            get_user_message_from_llama3_prompt(text) for text in batch[self.text_key]
        ]
        try:
            embeddings = self.model.encode(
                texts, convert_to_numpy=True, show_progress_bar=False
            )
            return {"id": ids, "embedding": embeddings}
        except Exception as e:
            logger.error(f"Error during embedding in actor: {e}", exc_info=True)
            # Return empty or partial data to avoid breaking pipeline? Or raise?
            # Let's return empty for now, downstream should handle potential empty batches.
            return {
                "id": np.array([], dtype=ids.dtype),
                "embedding": np.array([], dtype=np.float32),
            }


class LLMEmbedder:
    def __init__(self, model_name: str, text_key: str):
        # Load tokenizer and model onto GPU
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            model_name,
        ).to("cuda")
        self.has_cls = any(
            [
                base
                for base in ["bert", "roberta"]
                if base in type(self.model).__name__.lower()
            ]
        )

        self.model.eval()
        self.text_key = text_key
        logger.info(f"Embedder actor initialized with model {model_name} on GPU.")

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        ids = batch["id"]
        texts = batch[self.text_key].tolist()

        try:
            inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                if self.has_cls:
                    embeddings = outputs.last_hidden_state[:, 0, :]
                else:
                    last_hidden = (
                        outputs.last_hidden_state
                    )  # (batch, seq_len, hidden_size)
                    attention_mask = inputs["attention_mask"].unsqueeze(
                        -1
                    )  # (batch, seq_len, 1)
                    summed = (last_hidden * attention_mask).sum(dim=1)
                    counts = attention_mask.sum(dim=1)
                    embeddings = summed / counts  # mean pooling
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                embeddings = embeddings.detach().cpu().numpy()

            return {"id": ids, "embedding": embeddings}

        except Exception as e:
            logger.error(f"Error during embedding in actor: {e}", exc_info=True)
            return {
                "id": np.array([], dtype=ids.dtype),
                "embedding": np.array([], dtype=np.float32),
            }


class UmapTransformer:

    def __init__(self, model_path: str):
        import cuml

        # Load model onto the GPU assigned to this actor
        self.model = load_model(model_path)
        if not isinstance(self.model, cuml.UMAP):
            raise TypeError(f"Loaded object is not a cuml.UMAP model from {model_path}")
        logger.info(f"UmapTransformer actor initialized with model from {model_path}.")

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # Assumes 'id' and 'embedding' columns
        ids = batch["id"]
        embeddings = batch["embedding"]
        try:
            low_dim_embeddings = self.model.transform(embeddings)
            return {"id": ids, "umap_embedding": low_dim_embeddings}
        except Exception as e:
            logger.error(f"Error during UMAP transform in actor: {e}", exc_info=True)
            return {
                "id": np.array([], dtype=ids.dtype),
                "umap_embedding": np.array([], dtype=np.float32),
            }


class CumlClusteringPredictor:
    def __init__(self, model_path: str, model_type: str):
        """
        Args:
            model_path: Path to a saved cuML model.
            model_type: Name of the cuML class (e.g., "HDBSCAN", "KMeans").
        """
        import cuml

        self.model_path = model_path
        self.model_type = model_type
        self.model = load_model(model_path)

        expected_class = getattr(cuml.cluster, model_type, None)
        if expected_class is None or not isinstance(self.model, expected_class):
            raise TypeError(
                f"Loaded model is not a cuml.cluster.{model_type} instance from {model_path}"
            )

        logger.info(f"{model_type} predictor initialized with model from {model_path}")

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        import cuml

        ids = batch["id"]
        embeddings = batch["embedding"]
        try:
            if self.model_type == "HDBSCAN":
                labels, _ = cuml.cluster.hdbscan.approximate_predict(
                    self.model, embeddings
                )
            else:
                labels = self.model.predict(embeddings)

            if hasattr(labels, "get"):  # Convert from CuPy to NumPy if needed
                labels = labels.get()
            return {"id": ids, "cluster_label": labels}
        except Exception as e:
            logger.error(
                f"Error during {self.model_type} prediction: {e}", exc_info=True
            )
            return {
                "id": ids,
                "cluster_label": np.full(len(ids), -1, dtype=np.int32),
            }


# --- Ray Remote Fitting Tasks ---


@ray.remote(num_gpus=1)  # Removed memory request for simplicity, add back if needed
def fit_umap_on_worker_gpu(
    dataset_ref: ray.data.Dataset,
    n_components: int,  # Fit for one component count per task now
    output_path: str,  # Path to save the fitted model
    umap_fit_id: str,
    metric: str = "euclidean",
) -> None:  # Task saves the model directly
    """Fits UMAP on locally collected data and saves the model."""
    import cuml

    logger.info(
        f"UMAP Fitter Worker ({umap_fit_id}): Starting local data collection..."
    )
    local_embeddings = []
    ds_embeddings_only = dataset_ref.select_columns(["embedding"])
    # Use large batches for collection efficiency
    for block in ds_embeddings_only.iter_batches(batch_size=8192, batch_format="numpy"):
        local_embeddings.append(block["embedding"])
    if not local_embeddings:
        raise ValueError(f"UMAP Fitter ({umap_fit_id}): No embeddings collected.")
    embeddings_np = np.concatenate(local_embeddings, axis=0)
    logger.info(
        f"UMAP Fitter Worker ({umap_fit_id}): Collected {embeddings_np.shape[0]} points. Fitting UMAP for {n_components} components..."
    )

    umap_model = cuml.UMAP(
        n_components=n_components,  # Increase dimensions for clustering (try 10, 20, 50)
        n_neighbors=50,  # Balance local/global structure (try 30-100)
        min_dist=0.1,  # Keep default or slightly adjust
        metric="cosine",  # **CRITICAL**: Use cosine for text embeddings
        init="spectral",  # Good for preserving structure
        random_state=42,  # For reproducibility
        verbose=True,  # Keep verbose on
    ).fit(embeddings_np)

    logger.info(
        f"UMAP Fitter Worker ({umap_fit_id}): Fitting complete. Saving model to {output_path}..."
    )
    save_model(umap_model, output_path)  # Save from worker
    logger.info(f"UMAP Fitter Worker ({umap_fit_id}): Model saved.")


@ray.remote(num_gpus=1)
def fit_hdbscan_on_worker_gpu(
    dataset_ref: ray.data.Dataset,  # Expects dataset with 'umap_embedding' (clustering dim)
    output_path: str,  # Path to save the fitted model
    min_cluster_size: int = 500,
    min_samples: int = 10,
) -> None:  # Task saves the model directly
    """Collects low-dim data locally, fits HDBSCAN, and saves the model.

    https://docs.rapids.ai/api/cuml/stable/api/#hdbscan
    """
    import cuml

    logger.info(f"HDBSCAN Fitter Worker: Starting local data collection...")
    local_embeddings = []
    ds_embeddings_only = dataset_ref.select_columns(["embedding"])
    for block in ds_embeddings_only.iter_batches(batch_size=8192, batch_format="numpy"):
        local_embeddings.append(block["embedding"])
    if not local_embeddings:
        raise ValueError("HDBSCAN Fitter: No low-dim embeddings collected.")
    embeddings_np = np.concatenate(local_embeddings, axis=0)
    logger.info(
        f"HDBSCAN Fitter Worker: Collected {embeddings_np.shape[0]} points. Fitting HDBSCAN..."
    )

    clusterer = cuml.cluster.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True,  # MUST be True for approximate_predict later
        gen_min_span_tree=False,  # saves memory
        verbose=True,
        metric="euclidean",
        # reduce noisy points
        # cluster_selection_epsilon=0.1,
        # alpha=0.8,
    ).fit(embeddings_np)

    logger.info(
        f"HDBSCAN Fitter Worker: Fitting complete. Saving model to {output_path}..."
    )
    save_model(clusterer, output_path)  # Save from worker
    logger.info(f"HDBSCAN Fitter Worker: Model saved.")

    # labels_ is the cluster assignment per point (-1 = noise)
    labels = clusterer.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    # Optional: count points per cluster
    cluster_counts = collections.Counter(labels)
    top_clusters = cluster_counts.most_common(5)

    logger.info(f"HDBSCAN completed.")
    logger.info(f"Estimated number of clusters: {n_clusters}")
    logger.info(f"Number of noise points: {n_noise}")
    logger.info(f"Top 5 clusters (label, size): {top_clusters}")


@ray.remote(num_gpus=1)
def fit_kmeans_on_worker_gpu(
    dataset_ref: ray.data.Dataset,  # Expects dataset with 'umap_embedding' (clustering dim)
    output_path: str,  # Path to save the fitted model
    n_clusters: int = 50,  # Number of clusters to form
    max_iter: int = 300,  # Maximum number of iterations
    tol: float = 1e-4,  # Tolerance to declare convergence
    random_state: int = 42,  # Random state for reproducibility
) -> None:  # Task saves the model directly
    """Collects low-dim data locally, fits K-means, and saves the model.
    https://docs.rapids.ai/api/cuml/stable/api/#k-means
    """
    import cuml

    logger.info(f"K-means Fitter Worker: Starting local data collection...")
    local_embeddings = []
    ds_embeddings_only = dataset_ref.select_columns(["embedding"])
    for block in ds_embeddings_only.iter_batches(batch_size=8192, batch_format="numpy"):
        local_embeddings.append(block["embedding"])
    if not local_embeddings:
        raise ValueError("K-means Fitter: No low-dim embeddings collected.")
    embeddings_np = np.concatenate(local_embeddings, axis=0)
    logger.info(
        f"K-means Fitter Worker: Collected {embeddings_np.shape[0]} points. Fitting K-means..."
    )
    clusterer = cuml.cluster.KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        verbose=True,
        init="k-means++",  # Better initialization strategy
        n_init=1,  # GPU K-means typically converges well with single init
    ).fit(embeddings_np)
    logger.info(
        f"K-means Fitter Worker: Fitting complete. Saving model to {output_path}..."
    )
    save_model(clusterer, output_path)  # Save from worker
    logger.info(f"K-means Fitter Worker: Model saved.")
    # labels_ is the cluster assignment per point
    labels = clusterer.labels_
    # Compute inertia (within-cluster sum of squares)
    inertia = clusterer.inertia_
    # Optional: count points per cluster
    cluster_counts = collections.Counter(labels)
    top_clusters = cluster_counts.most_common(5)
    logger.info(f"K-means completed.")
    logger.info(f"Number of clusters: {n_clusters}")
    logger.info(f"Inertia: {inertia}")
    logger.info(f"Top 5 clusters (label, size): {top_clusters}")


def pandas_join(
    ds1: ray.data.Dataset,
    ds2: ray.data.Dataset,
    join_key: str,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """
    Workaround for Ray Dataset join using Pandas merge on a sampled subset.
    """
    if join_key not in ds1.schema().names or join_key not in ds2.schema().names:
        raise ValueError(f"Join key '{join_key}' not found in one or both datasets.")

    logger.info(f"Sampling {sample_size} rows from the first dataset...")
    ds1_count = ds1.count()
    if ds1_count == 0:
        logger.warning("First dataset is empty, returning empty dataset.")
        return ray.data.from_pandas(pd.DataFrame())

    if sample_size is not None:
        fraction = min(sample_size / ds1_count, 1.0) if ds1_count > 0 else 0
        sampled_ds1 = ds1.random_sample(fraction=fraction)
        sampled_df1 = sampled_ds1.to_pandas()
    else:
        sampled_df1 = ds1.to_pandas()

    logger.info("Collecting IDs from sampled dataset...")
    # to_pandas() brings data to the driver, assumes sample_size is manageable
    try:
        sample_ids = sampled_df1[join_key].unique().tolist()
    except Exception as e:
        logger.error(f"Error collecting sample IDs: {e}", exc_info=True)
        raise

    logger.info(f"Filtering second dataset by {len(sample_ids)} sampled IDs...")
    filtered_ds2 = ds2.filter(lambda row: row[join_key] in sample_ids)

    logger.info("Converting sampled and filtered data to Pandas for merge...")
    filtered_df2 = filtered_ds2.to_pandas()

    logger.info(f"Performing Pandas merge on '{join_key}'...")
    merged_df = pd.merge(
        sampled_df1, filtered_df2, on=join_key, how="inner"
    )  # Inner join matches Dataset.join default

    return ray.data.from_pandas(merged_df)


def summarize_clustering_stats(label_ds: ray.data.Dataset) -> None:
    """
    Calculates and logs key statistics from a Ray Dataset's 'cluster_label'
    column, assuming -1 represents noise.

    Args:
        label_ds: The input Ray Dataset expected to have at least a
                  'cluster_label' column.
    """
    logger.info("\n--- Cluster Stats ---")

    # 1. Total number of points
    total_rows = label_ds.count()
    logger.info(f"Total data points in dataset: {total_rows}")

    cluster_sizes_ds = label_ds.groupby("cluster_label").count()

    # The number of unique positive clusters is the number of rows in this result dataset
    n_clusters = cluster_sizes_ds.count()
    logger.info(f"Estimated number of clusters (including -1): {n_clusters}")

    histogram_raw_data: tp.List[tp.Dict] = cluster_sizes_ds.take_all()

    # Convert the collected data into a list of (label, size) tuples
    cluster_sizes_list: tp.List[tp.Tuple[int, int]] = []
    for row in histogram_raw_data:
        # The default column name for the count aggregation is 'count()'
        label = row["cluster_label"]
        size = row["count()"]
        cluster_sizes_list.append((label, size))

    # Sort the list of (label, size) tuples by size in descending order
    cluster_sizes_list.sort(key=lambda item: item[1], reverse=True)

    # Get the top 5 clusters
    top_clusters = cluster_sizes_list[:5]

    logger.info(f"Top 5 clusters (label, size): {top_clusters}")
    logger.info(f"Smallest 5 clusters (label, size): {cluster_sizes_list[-5:]}")

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    perc_values = np.percentile(
        np.array([s[1] for s in cluster_sizes_list]), percentiles
    )
    logger.info(f"Percentile sizes of clusters: {list(zip(percentiles, perc_values))}")

    logger.info("-----------------------------")


def inner_join_basic_types(
    left: ray.data.Dataset,
    right: ray.data.Dataset,
    key: str,
    left_prefix: str = "left_",
    right_prefix: str = "right_",
) -> ray.data.Dataset:
    """
        Use groupby to implement inner join.

    left = ray.data.from_items([
        {"id": 1, "value": "a"},
        {"id": 2, "value": "b"},
        {"id": 2, "value": "c"},
    ])

    right = ray.data.from_items([
        {"id": 2, "score": 100},
        {"id": 1, "score": 200},
        {"id": 3, "score": 300},
    ])

    joined = inner_join(left, right, key="id")
    print(joined.take_all())
    """

    left_cols = set(left.schema().names)
    right_cols = set(right.schema().names)
    shared_cols = left_cols & right_cols
    if shared_cols != {key}:
        raise ValueError(
            f"Expected only common column to be '{key}', but found: {shared_cols}"
        )

    print(left.schema, right.schema, left.count(), right.count())

    def normalize(ds: ray.data.Dataset, prefix: str, tag: str) -> ray.data.Dataset:
        renamed = ds.rename_columns(
            {col: f"{prefix}{col}" for col in ds.schema().names if col != key}
        )

        return renamed.add_column("__source__", lambda _: tag)

    left_tagged = normalize(left, left_prefix, "left")
    right_tagged = normalize(right, right_prefix, "right")
    combined = left_tagged.union(right_tagged)

    # map_groups gives each group as a list of rows
    def join_group(rows: pd.DataFrame) -> pd.DataFrame:
        key_value = rows.iloc[0][key]
        left_rows = (
            rows[rows["__source__"] == "left"]
            .drop(columns="__source__")
            .reset_index(drop=True)
        )
        right_rows = (
            rows[rows["__source__"] == "right"]
            .drop(columns="__source__")
            .reset_index(drop=True)
        )

        if left_rows.empty or right_rows.empty:
            return pd.DataFrame([])

        # Drop key from right rows to avoid duplicate column
        right_rows = right_rows.drop(columns=[key], errors="ignore")
        left_rows = left_rows.drop(columns=[key], errors="ignore")

        # Cartesian product (cross join) without suffix confusion
        joined = left_rows.merge(right_rows, how="cross")

        def extra_column(name):
            return (
                name.startswith("left_")
                and name.endswith("_y")
                or name.startswith("right_")
                and name.endswith("_x")
            )

        joined = joined.drop(
            columns=[col for col in joined.columns if extra_column(col)]
        )
        joined = joined.rename(
            columns=lambda col: re.sub(r"^(left_|right_)|(_x|_y)$", "", col)
        )

        joined.insert(0, key, key_value)

        return joined

    return combined.groupby(key).map_groups(join_group)


def inner_join(
    left: ray.data.Dataset,
    right: ray.data.Dataset,
    key: str,
    left_prefix: str = "left_",
    right_prefix: str = "right_",
) -> ray.data.Dataset:
    """
    Use groupby to implement inner join with simplified serialization approach.
    Only keeps the key column intact, and serializes all other columns into a single bytes column to handle complex data type such as numpy.ndarray.
    """
    left_cols = set(left.schema().names)
    right_cols = set(right.schema().names)
    shared_cols = left_cols & right_cols
    if shared_cols != {key}:
        raise ValueError(
            f"Expected only common column to be '{key}', but found: {shared_cols}"
        )

    print(left.schema, right.schema, left.count(), right.count())

    def serialize_row(batch, prefix):
        """Serialize all columns except the key into a single bytes column"""
        result = {key: batch[key]}
        data_list = []

        for _, row in batch.iterrows():
            # Extract all columns except key into a dictionary
            row_data = {col: row[col] for col in batch.columns if col != key}
            # Serialize the dictionary
            serialized = pickle.dumps(row_data)
            data_list.append(serialized)

        result[f"{prefix}data"] = data_list
        return pd.DataFrame(result)

    # Convert datasets to simplified form (key + serialized data)
    left_simple = left.map_batches(
        lambda batch: serialize_row(batch, left_prefix), batch_format="pandas"
    )
    right_simple = right.map_batches(
        lambda batch: serialize_row(batch, right_prefix), batch_format="pandas"
    )

    # Attach source tags
    left_tagged = left_simple.add_column("__source__", lambda _: "left")
    right_tagged = right_simple.add_column("__source__", lambda _: "right")

    # Combine the datasets
    combined = left_tagged.union(right_tagged)

    def join_group(rows: pd.DataFrame) -> pd.DataFrame:
        key_value = rows.iloc[0][key]

        # Extract left and right rows
        left_rows = (
            rows[rows["__source__"] == "left"]
            .drop(columns="__source__")
            .reset_index(drop=True)
        )
        right_rows = (
            rows[rows["__source__"] == "right"]
            .drop(columns="__source__")
            .reset_index(drop=True)
        )

        if left_rows.empty or right_rows.empty:
            return pd.DataFrame([])

        result_rows = []

        # Create cross join
        for _, left_row in left_rows.iterrows():
            left_data = pickle.loads(left_row[f"{left_prefix}data"])

            for _, right_row in right_rows.iterrows():
                right_data = pickle.loads(right_row[f"{right_prefix}data"])

                # Merge data
                new_row = {key: key_value}

                # Add prefixed columns from left
                for col, val in left_data.items():
                    new_row[col] = val

                # Add prefixed columns from right
                for col, val in right_data.items():
                    new_row[col] = val

                result_rows.append(new_row)

        if not result_rows:
            return pd.DataFrame([])

        return pd.DataFrame(result_rows)

    return combined.groupby(key).map_groups(join_group)

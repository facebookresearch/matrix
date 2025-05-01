import logging
import os
import time

# import argparse # Remove
import fire  # Add
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Basic Logging Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(
    input_dir: str,
    run_id: str = "0",
):

    viz_sample_path = os.path.join(input_dir, f"viz_{run_id}.pickle.df")
    output_plot_path = os.path.join(input_dir, f"viz_{run_id}.plot.png")

    viz_df = pd.read_pickle(viz_sample_path)
    print(viz_df.info())
    # --- 5. Generate Plot ---
    logger.info(f"Generating plot with {len(viz_df)} points...")
    try:
        # Extract data, handle numpy stacking robustly
        if "umap_viz" not in viz_df.columns or "cluster_label" not in viz_df.columns:
            raise ValueError(
                "Collected DataFrame missing 'umap_viz' or 'cluster_label' columns."
            )

        umap_viz_coords = np.stack(viz_df["umap_viz"].values)
        cluster_labels = viz_df["cluster_label"].values

        plt.figure(figsize=(14, 12))
        unique_labels = np.unique(cluster_labels)
        # Handle case where all points might be noise
        n_clusters = (
            len(unique_labels[unique_labels >= 0]) if np.any(unique_labels >= 0) else 0
        )
        # Use a colormap with enough distinct colors
        cmap = plt.cm.get_cmap(
            "tab20", max(1, len(unique_labels))
        )  # Ensure at least 1 color
        label_color_map = {label: cmap(i) for i, label in enumerate(unique_labels)}

        # Plot noise points first
        noise_mask = cluster_labels == -1
        if np.any(noise_mask):
            plt.scatter(
                umap_viz_coords[noise_mask, 0],
                umap_viz_coords[noise_mask, 1],
                s=2,
                c="grey",
                label="Noise (-1)",
                alpha=0.2,
            )

        # Plot clustered points
        for label in unique_labels:
            if label == -1:
                continue
            mask = cluster_labels == label
            if np.any(mask):  # Only plot if points exist for this label
                plt.scatter(
                    umap_viz_coords[mask, 0],
                    umap_viz_coords[mask, 1],
                    s=5,
                    color=label_color_map[label],
                    label=f"Cluster {label}",
                    alpha=0.6,
                )

        plt.title(f"HDBSCAN Clustering Visualization ({n_clusters} clusters found)")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.grid(True, linestyle="--", alpha=0.4)
        # Add legend only if few clusters, otherwise it gets cluttered
        if len(unique_labels) < 20:
            plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout if legend is outside
        plt.savefig(output_plot_path)
        logger.info(f"Plot saved to {output_plot_path}")

    except Exception as e:
        logger.error(f"Error during plot generation: {e}", exc_info=True)


if __name__ == "__main__":
    fire.Fire(main)

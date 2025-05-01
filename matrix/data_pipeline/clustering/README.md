pip install cuml-cu12 sentence_transformers pandas ray[data]

python fit.py \
    --input-jsonl "/home/dongwang/workspace/backup/text_cluster/cc_100k.jsonl" \
    --artifact-dir ./pipeline_artifacts \
    --umap-fit-sample-frac 0.1 \
    --hdbscan-fit-sample-frac 0.2 \
    --save-embeddings # Optional: if you want to save embeddings


# Replace 'YOUR_RUN_ID' with the actual ID from the fit step
python infer.py \
    --input-path "./pipeline_artifacts/embeddings_YOUR_RUN_ID.parquet" \
    --input-type embeddings \
    --artifact-dir ./pipeline_artifacts \
    --run-id YOUR_RUN_ID \
    --output-path ./pipeline_artifacts/inference_results_YOUR_RUN_ID.parquet \
    # Add resource/batch size args if needed


# Replace 'YOUR_RUN_ID' with the actual ID
python visualize.py \
    --input-path "./pipeline_artifacts/embeddings_YOUR_RUN_ID.parquet" \
    --input-type embeddings \
    --artifact-dir ./pipeline_artifacts \
    --inference-results-path ./pipeline_artifacts/inference_results_YOUR_RUN_ID.parquet \
    --run-id YOUR_RUN_ID \
    --output-plot-path ./cluster_visualization_YOUR_RUN_ID.png \
    --viz-sample-size 50000 \
    # Add resource/batch size args if needed
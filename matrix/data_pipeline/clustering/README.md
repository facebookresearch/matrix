# Matrix Embedding based Clustering

## Installation
```
pip install cuml-cu12~=25.4.0 sentence_transformers~=4.1.0 pandas~=2.2.3
```

## Examples

### K-means
- fit
```
python -m matrix.data_pipeline.clustering.fit $RAY_HEAD $INPUT_JSONL $OUTPUT_DIR --max_concurrency 16 --text_key '' --message_key "request.messages[0].content" --embedding_model princeton-nlp/sup-simcse-roberta-large --kmeans_num_clusters 100
```
- inference
```
python -m matrix.data_pipeline.clustering.inference $RAY_HEAD $INPUT_JSONL $OUTPUT_DIR --max_concurrency 16 --embedding_model princeton-nlp/sup-simcse-roberta-large --kmeans_num_clusters 100 
```

- sample
```
python -m matrix.data_pipeline.clustering.sample  $RAY_HEAD $OUTPUT_DIR --embedding_model princeton-nlp/sup-simcse-roberta-large --kmeans_num_clusters 100 --sample_size 10000
```

### HDBSCAM
- fit
```
python -m matrix.data_pipeline.clustering.fit $RAY_HEAD $INPUT_JSONL $OUTPUT_DIR --max_concurrency 16 --text_key '' --message_key "request.messages[0].content" --embedding_model meta-llama/Llama-3.1-8B-Instruct --enable_umap True --cluster_alg hdbscan --hdbscan_min_cluster_size 100
```

- inference
```
python -m matrix.data_pipeline.clustering.inference $RAY_HEAD $INPUT_JSONL $OUTPUT_DIR --max_concurrency 16 --embedding_model meta-llama/Llama-3.1-8B-Instruct --enable_umap True --cluster_alg hdbscan --hdbscan_min_cluster_size 100
```

- sample
```
python -m matrix.data_pipeline.clustering.sample  $RAY_HEAD $OUTPUT_DIR --max_concurrency 16 --embedding_model meta-llama/Llama-3.1-8B-Instruct --enable_umap True --cluster_alg hdbscan --hdbscan_min_cluster_size 100 --sample_size 10000
```
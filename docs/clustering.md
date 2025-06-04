# Matrix Embedding based Clustering

## Overview

Use semantic clustering to cluster and subsample text.

### Embedding
We tested three embedding models: all-MiniLM-L6-v2 (Wang et al., 2020), SimCSE (Gao et al., 2022) and Llama-3.1-8B-Instruct (Grattafiori et al., 2024). The embedding size of all-MiniLM-L6-v2 is 384, sup-simcse-roberta-large is 1024 and Llama-3.1-8B-Instruct is 4096. After the embedding is calculated, we can also reduce the dimension by projecting into a lower dimensional space through UMAP.

### Clustering Fit and Inference

With embeddings populated, we can fit a k-means or hdbscan clustering. The fit can be on a small subset to speed up the process. After the fitted model is obtained, it can be applied in parallel to the whole dataset.

### Sampling from Clusters

After the whole dataset is clustered, we can pick representative samplings based on the clusters. There are two options to control the subsample.
- `same_samples_per_cluster`: when it is True, each cluster will get similar number of samples; otherwise, larger clusters will get more samples proportional to the sqrt of the cluster size.
- `random_sample_within_cluster`: when it is True, random samples are chosen from a cluster; otherwise diverse samples that are far from each other in the embedding space are chosen.


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
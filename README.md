# Matrix: Multi-Agent daTa geneRation Infra and eXperimentation

Matrix is a versatile toolkit for synthetic data generation. It is the inference engine of [Collaborative Reasoner](https://github.com/facebookresearch/collaborative-reasoner) for multi-agent conversation generation.

## Features
Matrix runs on top of a Ray cluster. Cluster resources are acquired from Slurm or local through submitit. The main features are:
- Run large scale inference for huggingface LLMs using vllm and sglang.
- Proxy server to support Azure OpenAI, SageMaker, Gemini models.
- Code execution service as a wrapper of bubblewrap.
- Data pipelines for data quality filtering and classifications.

### Matrix vs. Existing Frameworks

Matrix is designed for scalable LLM inference on Slurm. Here is a feature comparison with other popular LLM inference solutions.


| Serving Frameworks | Slurm | vLLM | HTTP | gRPC | Auto-scaling | Open-source |
|-------------------|:-----:|:----:|:----:|:----:|:-----------:|:-----------:|
| vector-inference | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| litellm | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ |
| ollama | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ |
| SageMaker | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ |
| llm-swarm | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| Matrix | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |


## Getting Started

- Conda Environment
```
conda create --name matrix python=3.10
conda activate matrix
pip install 'git+ssh://git@github.com/facebookresearch/matrix.git#egg=matrix[vllm_073]'
```

- Launch ray cluster
```
matrix start_cluster --add_workers 1 --slurm "{'account': $SLURM_ACCOUNT, 'qos': $SLURM_QOS}"
```

- Deploy Model
```
matrix deploy_applications --applications "[{'model_name': 'meta-llama/Llama-3.1-8B-Instruct', 'min_replica': 8}]"
```

- LLM Inference
```
matrix check_health --app_name 8B
```

- Shudown ray cluster
```
matrix stop_cluster
```

## Advanced Deployment
### Enable Grafana Dashboard

- Install in conda
```
bash ./matrix/scripts/install_prometheus_and_grafana.sh
```
- Enable in Ray Dashboard
```
matrix start_cluster --enable_grafana
```

### Incremental Deployment

- Add More Workers
```
matrix start_cluster --add_workers 4 --slurm "{'account': $SLURM_ACCOUNT, 'qos': $SLURM_QOS}"
```

- Add/Remove Applications
```
matrix deploy_applications --action add --applications "[{'model_name': 'meta-llama/Llama-3.1-405B-Instruct', 'min_replica': 2}]"
```

- Remove All Applications
```
matrix deploy_applications --applications ''
```
### Adjust Context Length
The prompt token length + max_tokens <= max-model-len.
```
matrix deploy_applications --applications "[{'name': '405B', 'model_name': '/path_to_dir/Llama-3.1-405B-Instruct', 'min_replica': 1, 'model_size': '405B', 'pipeline-parallel-size': 4, 'max-model-len': 30960}]"
```

405B recommended parameters
| Context | Parallel |
| ------- | -------- |
| 10k | 2 |
| 30k | 4 |
| 60k | 6 |
| 128k | 10 |

### OpenAI Azure Model
- Note: no GPU is required, in start_workers, can add `--slurm "{'gpus_per_node': 0}"`

```
matrix deploy_applications --applications "[{'api_version': \"$AZURE_API_VERSION\", 'api_endpoint': \"$AZURE_ENDPOINT\", 'api_key': \"$AZURE_API_KEY\", 'app_type': 'openai', 'model_name': 'gpt-4o', 'name': 'openai'}]"
```

### Gemini
- Note: no GPU is required, in start_workers, can add `--slurm "{'gpus_per_node': 0}"`

```
matrix deploy_applications --applications "[{'app_type': 'gemini', 'name': "gemini", 'api_key': \"$GOOGLE_API_KEY\",  'model_name': 'gemini-2.0-flash'}]"
```

### Deepseek R1
```
// install sglang
pip install 'git+ssh://git@github.com/facebookresearch/matrix.git#egg=matrix[sglang_043]'

matrix deploy_applications --applications "[{'model_name': 'deepseek-ai/DeepSeek-R1', 'min_replica': 2, 'app_type': sglang_llm}]"
```
### Llama 4
```
pip install 'git+ssh://git@github.com/facebookresearch/matrix.git#egg=matrix[vllm_083]'

matrix deploy_applications --applications "[{'model_name': 'meta-llama/Llama-4-Scout-17B-16E-Instruct'}]"

matrix deploy_applications --applications "[{'model_name': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8'}]"
```

## LLM Inference
### Batch Query
```
// download math-500 dataset
python -m matrix.scripts.hf_dataset_to_jsonl HuggingFaceH4/MATH-500 test test.jsonl

// query math-500
matrix llm_inference --app_name maverick-fp8 --input_jsonls test.jsonl --output_jsonl response.jsonl --batch_size=64 --system_prompt "Please reason step by step, and put your final answer within \boxed{}." --max_tokens 30000 --text_key problem --timeout_secs 1800
```

#### Input Format
There are two format for the jsonl input files:
  - Message foramt with arg --messages_key request.messages
```json
{
    "request": {"messages": [{"role": "system","content": "You are ..."},{"role": "user","content": "Solve the following..."}]}
}
```
  - Instruct format with arg --text_key text
```json
{
    "text": "<|start_header_id|>system<|end_header_id|>You are ... <|eot_id|><|start_header_id|>user<|end_header_id|>Solve the following ...<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
}
```
  - Raw tax with arg --text_key text
```json
{
    "text": "Solve the following ..."
}
```
### Inference API
```
from matrix import Cli
from matrix.app_server.llm import query_llm

metadata = Cli().get_app_metadata(app_name="8B")
await query_llm.make_request(
  url=metadata["endpoints"]["head"],
  model=metadata["model_name"],
  app_name=metadata["name"],
  data={"messages": [{"role": "user", "content": "hi"}]},
))
```

## Code Execution
- Install bubblewrap
```
conda install -c conda-forge bubblewrap
```
- Run example python code
```
matrix deploy_applications --applications "[{'name': 'code', 'app_type': code}]"
matrix check_health --app_name code
```

## Data Pipelines
- minhash dedup
```
python matrix/data_pipeline/quality/dedup_minhash.py ray_head:client_server_port input.jsonl output_dir working_dir
```
- multilabel classification
```
python matrix/data_pipeline/classification/multi_label_classification.py ray_head:client_server_port  cardiffnlp/twitter-roberta-base-emotion-multilabel-latest input.jsonl output_dir --num_gpus 48 --text_key question --threshold_fname ""
```

## Contributing
We always welcome contributions to matrix! Please refer to
[Contribution Guidelines](CONTRIBUTING.md) to learn how to format, test, and
submit your work.


## Citing Matrix
If you use matrix in your research and wish to refer to it, please use the
following BibTeX entry.

```
@software{wang2025matrix,
  author = {Dong Wang and Yang Li and Ansong Ni and Youssef Emad and Xinjie Lei and Ruta Desai and Asli Celikyilmaz and Daniel Li},
  title = {Matrix},
  url = {http://github.com/facebookresearch/matrix},
  year = {2025},
}
```


## License
This project is MIT licensed, as found in the [LICENSE](LICENSE) file.

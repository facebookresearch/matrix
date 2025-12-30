<h1 align="center">
Matrix: Multi-Agent daTa geneRation Infra and eXperimentation
</h1>

<h3 align="center">
Fast, scalable, and easy-to-use LLM-generation engine
</h3>

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Lint and Tests](https://github.com/facebookresearch/matrix/actions/workflows/lint_and_tests.yml/badge.svg)](https://github.com/facebookresearch/matrix/actions/workflows/lint_and_tests.yml?query=branch%3Amain)
[![Test Docker Image](https://github.com/facebookresearch/matrix/actions/workflows/docker_image.yml/badge.svg)](https://github.com/facebookresearch/matrix/actions/workflows/docker_image.yml?query=branch%3Amain)
[![License](https://img.shields.io/badge/license-MIT-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](./LICENSE)

</div>

---

*Latest News*
* 11/2025: paper [Matrix: Peer-to-Peer Multi-Agent Synthetic Data Generation Framework](https://arxiv.org/abs/2511.21686) is released.
* 04/2025: 🔥 We officially released Matrix with [Collaborative Reasoner](https://github.com/facebookresearch/collaborative-reasoner), showcasing the generation of multi-agent collaborative conversation with Matrix as inference engine. 

---

# About

Matrix is a library for fast, scalable, and easy-to-use LLM-generation engine, for use cases including model benchmarking, data processing, and data generation. 

Matrix runs on top of a [Ray](https://github.com/ray-project/ray) cluster for scalability. Cluster resources are acquired from [Slurm](https://slurm.schedmd.com/documentation.html) or local through [submitit](https://github.com/facebookincubator/submitit). Matrix has following main features:

**Large scale inference** for maintstream opensourced and proprietary LLMs
- Hugging Face LLMs via seamless integration with [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang). Native multi-node inference support.
- Azure OpenAI, SageMaker, Gemini models with Proxy server

**Data pipelines** of high-throughput data processing and quality checks
- Code execution service as a wrapper of [bubblewrap](https://github.com/containers/bubblewrap).
- Data curation, quality filtering, and augmentation with classifiers.

**Peer-to-peer** multi-agent orchestration
- 2-15× higher throughput.
- 10,000s of concurrent workflows.
- Adapts to diverse generation tasks.

### Matrix vs. Existing Frameworks

Matrix is designed for scalable LLM inference on [Slurm](https://slurm.schedmd.com/documentation.html). Here is a feature comparison with other popular LLM inference solutions.


| Serving Frameworks | Slurm | vLLM | HTTP | gRPC | Auto-scaling | Open-source |
|-------------------|:-----:|:----:|:----:|:----:|:-----------:|:-----------:|
| vector-inference | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| litellm | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ |
| ollama | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ |
| SageMaker | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ |
| llm-swarm | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| Matrix | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## Quick Links
  - [Getting Started](#getting-started)
  - [Advanced Deployment](#advanced-deployment)
  - [LLM Inference](#llm-inference)
  - [Job Manager](#job-manager)
  - [Data pipelines](#data-pipelines)
  - [Peer-to-peer](#peer-to-peer)
  - [Contributing](#contributing)
  - [Citation](#citation)

---

## Getting Started

- Conda Environment
```bash
conda create --name matrix python=3.11
conda activate matrix
pip install fair-matrix[vllm_0112]
```

- Launch ray cluster
```bash
matrix start_cluster --add_workers 1 --slurm "{'account': $SLURM_ACCOUNT, 'qos': $SLURM_QOS}"
```

- Deploy Model
```bash
# login to access huggingface hub
huggingface-cli login

matrix deploy_applications --applications "[{'model_name': 'meta-llama/Llama-3.1-8B-Instruct', 'min_replica': 8, 'name': '8B'}]"
```

- LLM Inference
```bash
matrix check_health --app_name 8B
```

- Shudown ray cluster
```bash
matrix stop_cluster
```

---

## Advanced Deployment
### Enable Grafana Dashboard

- Install in conda
```bash
bash ./matrix/scripts/install_prometheus_and_grafana.sh
```
- Enable in Ray Dashboard
```bash
matrix start_cluster --enable_grafana
```

### Incremental Deployment

- Add More Workers
```bash
matrix start_cluster --add_workers 4 --slurm "{'account': $SLURM_ACCOUNT, 'qos': $SLURM_QOS}"
```

- Add/Remove Applications
```bash
matrix deploy_applications --action add --applications "[{'model_name': 'meta-llama/Llama-3.1-405B-Instruct', 'min_replica': 2, 'name': '405B'}]"
```

- Remove All Applications
```bash
matrix deploy_applications --applications ''
```
### Adjust Model Args
vLLM Engine [Arguments](https://docs.vllm.ai/en/latest/serving/engine_args.html) can be specified in the deploy_applications arguments. The default values for popular models are in [llm_config.py](matrix/app_server/llm/llm_config.py). Other useful args
* `model_name`: a huggingface model name or a directory containing checkpoints.
* `name`: the default app_name.
* `model_size`: template to apply when model is from a directory, such as 8B, 70B, 405B etc, templates are from the llm_config.py file.
* `max_ongoing_requests`: the max concurrent requests to each replica.
* `min_replia` and `max_replica`: the num of replicas ranges auto-scaled based on num of Ray workers.
* `use_grpc`: enable grpc by adding `{'use_grpc':  'true'}`.

### OpenAI Azure Model
- Note: no GPU is required, in start_workers, can add `--slurm "{'gpus_per_node': 0}"`

```bash
matrix deploy_applications --applications "[{'api_version': \"$AZURE_API_VERSION\", 'api_endpoint': \"$AZURE_ENDPOINT\", 'api_key': \"$AZURE_API_KEY\", 'app_type': 'openai', 'model_name': 'gpt-4o', 'name': 'openai'}]"
```

### Gemini
- Note: no GPU is required, in start_workers, can add `--slurm "{'gpus_per_node': 0}"`

```bash
matrix deploy_applications --applications "[{'app_type': 'gemini', 'name': "gemini", 'api_key': \"$GOOGLE_API_KEY\",  'model_name': 'gemini-2.0-flash'}]"
```

### Deepseek R1
vLLM >=0.8.3 supports DS R1. An alternative backend is sglang.
```bash
# install sglang
pip install fair-matrix[sglang_045]

matrix deploy_applications --applications "[{'model_name': 'deepseek-ai/DeepSeek-R1', 'pipeline-parallel-size': 2, 'app_type': sglang_llm, 'name': 'r1'}]"
```
### Llama 4
```bash
matrix deploy_applications --applications "[{'model_name': 'meta-llama/Llama-4-Scout-17B-16E-Instruct', 'name': 'scout'}]"

matrix deploy_applications --applications "[{'model_name': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8', 'name': 'maverick'}]"
```

---

## LLM Inference

### Batch Query
```bash
# download math-500 dataset
python -m matrix.scripts.hf_dataset_to_jsonl HuggingFaceH4/MATH-500 test test.jsonl

# query math-500 from local jsonl
matrix inference --app_name maverick-fp8 --input_jsonls test.jsonl --output_jsonl response.jsonl --batch_size=64 \
  --system_prompt "Please reason step by step, and put your final answer within \boxed{}." --max_tokens 30000 --text_key problem --timeout_secs 1800

# or query directly from the Hugging Face dataset
matrix inference --app_name maverick-fp8 --input_hf_dataset HuggingFaceH4/MATH-500 --hf_dataset_split test \
  --output_jsonl response.jsonl --batch_size=64 \
  --system_prompt "Please reason step by step, and put your final answer within \boxed{}." --max_tokens 30000 --text_key problem --timeout_secs 1800
```

#### Input Format
There are two formats for the jsonl input files:
  - Message format with arg --messages_key request.messages
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
  - Raw text with arg --text_key text
```json
{
    "text": "Solve the following ..."
}
```
### Inference API
```python
from matrix import Cli
from matrix.client import query_llm
import asyncio

metadata = Cli().get_app_metadata(app_name="8B")

# async call
resp = asyncio.run(query_llm.make_request(
  url=metadata["endpoints"]["head"],
  model=metadata["model_name"],
  app_name=metadata["name"],
  data={"messages": [{"role": "user", "content": "hi"}]},
))
print(resp)

# batch inference
for response in query_llm.batch_requests(
    url=metadata["endpoints"]["head"],
    model=metadata["model_name"],
    app_name=metadata["name"],
    requests=[{"messages": [{"role": "user", "content": "hi"}]}],
):
    print(response)
```

#### Running inside Docker

To execute the same snippet within the Matrix Docker image, run:
```bash
# build the image with vLLM support (default)
docker build -t matrix .

# inference
docker run --rm \
  --network=host \
  --entrypoint python \
  -e MATRIX_CLUSTER_ID="${USER}_cluster" \
  -v ~/.matrix:/home/appuser/.matrix:ro \
  matrix -u - <<'EOF'
from matrix import Cli
from matrix.client import query_llm
import asyncio

metadata = Cli().get_app_metadata(app_name="8B")

# async call
resp = asyncio.run(query_llm.make_request(
  url=metadata["endpoints"]["head"],
  model=metadata["model_name"],
  app_name=metadata["name"],
  data={"messages": [{"role": "user", "content": "hi"}]},
))
print(resp)

# batch inference
for response in query_llm.batch_requests(
    url=metadata["endpoints"]["head"],
    model=metadata["model_name"],
    app_name=metadata["name"],
    requests=[{"messages": [{"role": "user", "content": "hi"}]}],
):
    print(response)
EOF
```
**NOTE:**
- If you are using custom cluster_id, update here `-e MATRIX_CLUSTER_ID="custom_cluster_id"`.
- If required export `HUGGING_FACE_HUB_TOKEN` on the host (and pass `-e HUGGING_FACE_HUB_TOKEN=...` to docker when needed).

---

## Job manager

Job manager allows users to submit tasks for distributed execution on Ray. More details are in [here](matrix/job/README.md).

---

## Data pipelines

### Code Execution
- Install bubblewrap
```bash
conda install -c conda-forge bubblewrap
```
- Run example python code
```bash
matrix deploy_applications --applications "[{'name': 'code', 'app_type': code, 'min_replica': 5}]"
matrix check_health --app_name code

python -m matrix.scripts.hf_dataset_to_jsonl openai/openai_humaneval test humaneval/test.jsonl
matrix inference code ~/tmp/he.jsonl humaneval/test.jsonl --text_keys "[prompt, canonical_solution, test, entry_point]" --prompt_template "check({entry_point})"
```

### Data filtering and augmentation
- minhash dedup
```bash
python -m matrix.data_pipeline.quality.dedup_minhash $ray_head:$client_server_port input.jsonl output_dir working_dir --text_key problem
```
- multilabel classification
```bash
python -m matrix.data_pipeline.classification.multi_label_classification $ray_head:$client_server_port  \
  cardiffnlp/twitter-roberta-base-emotion-multilabel-latest input.jsonl output_dir \
  --num_gpus 8 --text_key question --threshold_fname ""
```
- Offline batch inference
```bash
python -m matrix.data_pipeline.generate.vllm_generate $ray_head:$client_server_port ./math-500/test.jsonl math-500/response  \
  --prompt_template "<|start_header_id|>system<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \boxed{}.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<user_message><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" \
  --model_args "{'model': 'meta-llama/Llama-3.3-70B-Instruct', 'seed': 42, 'max_model_len': 20480, 'tensor_parallel_size': 4}" \
  --sampling_params "{'n': 1, 'temperature': 0.6, 'top_p': 0.95, 'max_tokens': 16384}" \
  --min_concurrency 32 --output_key pred --batch_size=32
```
---

## Peer-to-peer

Peer-to-peer framework avoids the single orchestration botttleneck and supports diverse synthetic data generaion tasks. More details are in [here](matrix/agents/README.md).

---

## Contributing
We always welcome contributions to matrix! Please refer to
[Contribution Guidelines](CONTRIBUTING.md) to learn how to format, test, and
submit your work. If you have any questions related to the code, 
feel free to email Dong Wang (dongwang@meta.com) or Daniel Li (shangwel@meta.com).

## Citation
If you use matrix in your research and wish to refer to it, please use the
following BibTeX entry.

```bibtex
@misc{wang2025matrixpeertopeermultiagentsynthetic,
      title={Matrix: Peer-to-Peer Multi-Agent Synthetic Data Generation Framework},
      author={Dong Wang and Yang Li and Ansong Ni and Ching-Feng Yeh and Youssef Emad and Xinjie Lei and Liam Robbins and Karthik Padthe and Hu Xu and Xian Li and Asli Celikyilmaz and Ramya Raghavendra and Lifei Huang and Carole-Jean Wu and Shang-Wen Li},
      year={2025},
      eprint={2511.21686},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.21686},
}

@software{matrix2025,
  author = {Dong Wang and Yang Li and Ansong Ni and Youssef Emad and Xinjie Lei and Ruta Desai and Karthik Padthe and Xian Li and Asli Celikyilmaz and Ramya Raghavendra and Leo Huang and Daniel Li},
  title = {Matrix: Multi-Agent daTa geneRation Infra and eXperimentation},
  url = {http://github.com/facebookresearch/matrix},
  year = {2025},
}
```

## License
This project is MIT licensed, as found in the [LICENSE](LICENSE) file.


## Acknowledgement
We gratefully acknowledge the [Ray](https://github.com/ray-project/ray) and [vLLM](https://github.com/vllm-project/vllm) team for initial Ray Serve integration with vLLM.

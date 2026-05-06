---
name: matrix-inference
description: Run LLM inference against deployed models - CLI batch, Python API, and multiprocessing
user-invocable: true
---

# Matrix LLM Inference

Help the user run inference against models deployed on their Matrix cluster.

## Prerequisites

- A running Matrix cluster with at least one model deployed
- Verify with `matrix check_health --app_name <name>` (this also does a basic inference test)

## Option 1: CLI Batch Inference

Best for processing JSONL files or HuggingFace datasets end-to-end.

Ask the user for:
- `app_name`: the deployed model's name
- Input source: `--input_jsonls <file.jsonl>` or `--input_hf_dataset <dataset> --hf_dataset_split <split>`
- `--output_jsonl`: output file path
- `--batch_size`: concurrent requests (default 128)
- `--max_tokens`: max generation tokens
- `--timeout_secs`: per-request timeout (default 600)
- `--max_retries`: retry count (default 3, use 100+ for long runs)
- `--text_key`: which field in the input JSON contains the prompt (e.g., `problem`, `question`)
- `--system_prompt`: optional system prompt
- `--messages_key`: alternative to text_key, for pre-formatted message arrays (e.g., `request.messages`)

Example:
```bash
matrix inference --app_name qwen3vl \
    --input_jsonls test.jsonl \
    --output_jsonl qwen3vl_response.jsonl \
    --batch_size=128 \
    --system_prompt "Please reason step by step, and put your final answer within \boxed{}." \
    --max_tokens 60000 \
    --text_key problem \
    --timeout_secs 3600 \
    --max_retries 100
```

From HuggingFace:
```bash
matrix inference --app_name 8B \
    --input_hf_dataset HuggingFaceH4/MATH-500 --hf_dataset_split test \
    --output_jsonl response.jsonl \
    --batch_size=64 \
    --system_prompt "Please reason step by step, and put your final answer within \boxed{}." \
    --max_tokens 30000 \
    --text_key problem \
    --timeout_secs 1800
```

### Input Formats

Three formats supported in JSONL:
- **Raw text** with `--text_key text`: `{"text": "Solve the following ..."}`
- **Messages** with `--messages_key request.messages`: `{"request": {"messages": [{"role": "user", "content": "..."}]}}`
- **Llama instruct format** with `--text_key text`: uses `<|start_header_id|>` tags, auto-parsed into messages

## Option 2: Python API (async, single process)

Best for programmatic use, notebooks, and integration into pipelines.

### Simple: `generate()` (highest-level)

```python
from matrix import Cli
from matrix.client.query_llm import generate

cli = Cli()
results = generate(
    cli,
    app_name="8B",
    prompts=[{"messages": [{"role": "user", "content": "hi"}]}],
    sampling_params={"temperature": 0.7, "max_tokens": 512},
    batch_size=128,
    max_retries=1000,
    text_response_only=True,  # returns list of strings
)
```

### Direct: `make_request()` (single async request)

```python
from matrix import Cli
from matrix.client import query_llm
import asyncio

cli = Cli()
metadata = cli.get_app_metadata(app_name="8B")

resp = asyncio.run(query_llm.make_request(
    url=metadata["endpoints"]["head"],
    model=metadata["model_name"],
    app_name=metadata["name"],
    data={"messages": [{"role": "user", "content": "hi"}]},
))
print(resp["response"]["text"])
```

### Batch (sync): `batch_requests()`

Sync wrapper that works from both sync and async contexts. Internally calls `batch_requests_async()` via `run_async()`.

```python
from matrix import Cli
from matrix.client.query_llm import batch_requests

cli = Cli()
metadata = cli.get_app_metadata(app_name="8B")

results = batch_requests(
    url=metadata["endpoints"]["head"],
    model=metadata["model_name"],
    app_name=metadata["name"],
    requests=[{"messages": [{"role": "user", "content": msg}]} for msg in prompts],
    batch_size=64,
    temperature=0.7,
    max_tokens=1024,
)
```

### Batch (async): `batch_requests_async()`

Use this when you're already in an async context. It uses `asyncio.Semaphore(batch_size)` for concurrency control and preserves input order.

```python
from functools import partial
from matrix.client.query_llm import make_request
from matrix.utils.os import batch_requests_async

results = await batch_requests_async(
    func=partial(make_request, url, model, **kwargs),
    args_list=[{"data": req} for req in requests],
    batch_size=128,
)
```

### Using EndpointCache (recommended for high throughput)

The head node is a network bottleneck. `EndpointCache` discovers all healthy Ray Serve worker nodes via the Ray dashboard API and routes requests directly to workers, bypassing the head. It refreshes the worker list on a TTL basis with double-checked locking.

```python
from matrix import Cli
from matrix.client.query_llm import batch_requests

cli = Cli()
metadata = cli.get_app_metadata(app_name="8B")

# Use url=None with endpoint_cache to route directly to workers
results = batch_requests(
    url=None,
    model=metadata["model_name"],
    app_name=metadata["name"],
    requests=my_requests,
    batch_size=128,
    endpoint_cache=metadata["endpoints"]["updater"],
)
```

`get_app_metadata()` automatically creates the `EndpointCache` at `metadata["endpoints"]["updater"]`. Pass `url=None` and `endpoint_cache=metadata["endpoints"]["updater"]` so requests go directly to workers.

## Option 3: Multiprocessing (large-scale inference)

Best for very large datasets where you need multiple processes each with their own async event loop. Uses `multiprocessing.Queue` for producer-consumer pattern.

```python
from matrix.client.llm_client import LLMClient

client = LLMClient(app_name="8B")

# data_loader is an Iterator[dict] where each dict has "messages" or "prompt"
data = [{"messages": [{"role": "user", "content": p}]} for p in prompts]

client.multiprocess_inference(
    data_loader=iter(data),
    task_params={"temperature": 0.7, "max_tokens": 1024},
    output_filepath="output.jsonl",
    n_process=4,                # number of worker processes
    n_request_per_process=1000, # concurrent requests per process
)
```

This gives `n_process * n_request_per_process` total concurrent requests. Each process independently resolves endpoints through `EndpointCache`, distributing load across all worker nodes. A separate collector process writes results to JSONL.

You can supply a custom `postproc_func(response_queue, output_filepath)` for custom output handling.

## Which Option to Choose

| Scenario | Recommended |
|----------|-------------|
| Process a JSONL/HF dataset from command line | CLI `matrix inference` |
| Programmatic, moderate scale (<10K requests) | `generate()` or `batch_requests()` |
| Very large scale (>10K requests) | `LLMClient.multiprocess_inference()` |
| Single request in application code | `make_request()` |
| High throughput, avoid head bottleneck | Use `EndpointCache` with any API option |

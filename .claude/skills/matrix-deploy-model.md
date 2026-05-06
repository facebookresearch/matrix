---
name: matrix-deploy-model
description: Deploy LLM or other applications on a Matrix Ray cluster
user-invocable: true
---

# Deploy Model Applications

Help the user deploy models and services on their Matrix cluster.

## Steps

1. Make sure the cluster is running:
```bash
matrix status
```

2. Ask the user what they want to deploy. Common options:

   **a) HuggingFace LLM** (most common)
   - `model_name`: HuggingFace model ID or local checkpoint path
   - `name`: short app name for referencing later
   - `min_replica`: number of replicas (default: 1)
   - `model_size`: template to use when loading from a local directory (e.g., `8B`, `70B`, `405B`) - templates are in `matrix/app_server/llm/llm_config.py`

   **b) Omni LLM** (image/audio generation)
   - Same as above plus `app_type: omni_llm`

   **c) Azure OpenAI**
   - Requires: `api_version`, `api_endpoint`, `api_key`, `app_type: openai`

   **d) Gemini**
   - Requires: `api_key`, `model_name`, `app_type: gemini`

   **e) Code execution**
   - `app_type: code`, `name: code`

3. Build and run the deploy command. Examples:

HuggingFace model:
```bash
matrix deploy_applications --applications "[{'model_name': 'meta-llama/Llama-3.1-8B-Instruct', 'min_replica': 8, 'name': '8B'}]"
```

Local checkpoint with omni:
```bash
matrix deploy_applications --action add --applications "[{'model_name': '/checkpoint/path/to/model', 'model_size': 'Qwen3-Omni-30B-A3B', 'name': 'qwen3', 'min_replica': 1, 'app_type': 'omni_llm'}]"
```

Multiple models at once:
```bash
matrix deploy_applications --applications "[{'model_name': 'meta-llama/Llama-4-Scout-17B-16E-Instruct', 'name': 'scout'}, {'model_name': 'meta-llama/Llama-3.1-8B-Instruct', 'min_replica': 1, 'name': '8B'}]"
```

4. Use `--action add` to add applications to existing ones, or omit for replace (default).

5. Verify deployment:
```bash
matrix check_health --app_name <name>
```

## Useful Options

- `use_grpc`: enable gRPC with `'use_grpc': 'true'`
- `max_ongoing_requests`: max concurrent requests per replica
- `min_replica` / `max_replica`: auto-scaling range based on Ray workers
- `enable_tools`: enable tool calling with `'enable_tools': 'true'`
- `pipeline-parallel-size`: for multi-node inference (e.g., large models)

## Remove All Applications

```bash
matrix deploy_applications --applications ''
```

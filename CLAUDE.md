# Matrix: Multi-Agent daTa geneRation Infra and eXperimentation

## Project Overview

Matrix is a library for fast, scalable LLM inference and multi-agent synthetic data generation, built on Ray and Slurm. It supports vLLM/SGLang-based inference, auto-scaling, gRPC/HTTP, and peer-to-peer multi-agent orchestration.

## Development

- Python 3.10+ required (3.12 recommended for new environments)
- Package: `fair-matrix`, installed via `uv pip install -e .[extra]`
- Build system: flit
- Linting: `black`, `isort`, `mypy`, `pylint`
- Tests: `pytest` (with `pytest-asyncio`)
- Format before committing: `black . && isort .`
- Entry point: `matrix` CLI defined in `matrix/cli.py`

## Install Extras

| Extra | Use Case |
|-------|----------|
| (none) | Client-only usage (query existing cluster) |
| `ray` | Launch cluster without LLM inference engine |
| `vllm_0112` | Most stable vLLM for LLM inference |
| `vllm_omni` | Omni vLLM with image/audio generation |
| `sglang_045` | SGLang backend (e.g. DeepSeek R1) |
| `agent` | Peer-to-peer agent framework (hydra, langgraph) |
| `dev` | Development tools (test, lint, docs) |

Other vLLM versions: `vllm_066`, `vllm_073`, `vllm_083`, `vllm_085`, `vllm_0102`

## Key Commands

```bash
matrix start_cluster    # Launch Ray cluster on Slurm
matrix stop_cluster     # Shutdown Ray cluster
matrix status           # Check cluster status, get ports for dashboards
matrix deploy_applications  # Deploy/remove model applications
matrix check_health     # Health check a deployed app
matrix inference        # Batch inference
```

## Architecture

- `matrix/cli.py` - CLI entry point
- `matrix/app_server/` - Application servers (LLM, code execution, etc.)
- `matrix/app_server/llm/llm_config.py` - Default vLLM configs per model size
- `matrix/client/` - Client library for querying deployed models
- `matrix/agents/` - Peer-to-peer multi-agent framework
- `matrix/agents/config/` - Hydra configs for agent tasks
- `matrix/data_pipeline/` - Data processing pipelines
- `matrix/job/` - Job manager for distributed execution
- `matrix/scripts/` - Utility scripts

## Skills

Use `/matrix-setup-conda` to create a conda environment and install matrix.
Use `/matrix-launch-cluster` to launch a Ray cluster on Slurm.
Use `/matrix-deploy-model` to deploy model applications on the cluster.
Use `/matrix-inference` to run LLM inference (CLI batch, Python API, or multiprocessing).
Use `/matrix-run-agents` to run peer-to-peer synthetic data generation.

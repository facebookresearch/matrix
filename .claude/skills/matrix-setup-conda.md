---
name: matrix-setup-conda
description: Create a conda environment and install fair-matrix with the appropriate extras
user-invocable: true
---

# Setup Conda Environment for Matrix

Help the user create a conda environment and install fair-matrix.

## Steps

1. Ask the user which install profile they need:
   - **Client only** (no extra) - just querying an existing Matrix cluster
   - **Cluster management** (`ray`) - launching/managing clusters without an LLM engine
   - **LLM inference** (`vllm_0112`) - most stable vLLM for running LLM inference
   - **Omni inference** (`vllm_omni`) - vLLM with image/audio generation support
   - **SGLang** (`sglang_045`) - SGLang backend (e.g., DeepSeek R1)
   - **Agent framework** (`agent`) - peer-to-peer multi-agent orchestration
   - **Development** (`dev`) - for contributing to matrix

2. Ask the user what Python version they want (default: 3.12).

3. Ask the user what environment name they want (default: `matrix`).

4. Generate and run the commands:

```bash
conda create -n <env_name> python=<python_version> pip uv -c conda-forge -y
conda activate <env_name>
```

5. Then install matrix. If the user is developing matrix locally (has the repo cloned):
```bash
uv pip install -e ".[<extra>]"
```

If installing from PyPI:
```bash
pip install "fair-matrix[<extra>]"
```

Multiple extras can be combined: `uv pip install -e ".[vllm_0112,agent,dev]"`

6. Verify the installation:
```bash
matrix --help
```

## Notes
- Use `uv pip install` for faster installs when installing from source
- For omni inference, `soundfile` is included automatically via the `vllm_omni` extra
- The `agent` extra adds `hydra-core`, `langgraph`, and `zstandard`

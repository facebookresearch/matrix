---
name: matrix-run-agents
description: Run peer-to-peer multi-agent synthetic data generation tasks
user-invocable: true
---

# Run Peer-to-Peer Agent Tasks

Help the user run multi-agent synthetic data generation using the Matrix P2P framework.

## Prerequisites

- Matrix installed with `agent` extra: `uv pip install -e ".[vllm_0112,agent]"`
- A running Matrix cluster with models deployed
- The deployed model app names must match what the agent config expects

## Steps

1. Ask the user which task they want to run. Available configs are in `matrix/agents/config/`:

   **a) Omni Debate** (`omni_debate.yaml`)
   - Two LLM agents debate topics with speech mode
   - Requires an omni model deployed (e.g., `qwen3` with `app_type: omni_llm`)

   **b) Collaborative Reasoner / CoRAL** (`coral_mmlu_pro.yaml`)
   - Student-teacher collaborative reasoning
   - Requires: `student_llm`, `teacher_llm`, `extractor_llm` services

   **c) Tau2-Bench** (`tau2_bench.yaml`)
   - Agent benchmark with containers
   - Requires building the apptainer image first

   **d) SWE-Bench** (`mini_swe_agent.yaml`)
   - Software engineering benchmark
   - Requires building apptainer images first

   **e) NR Curation** (`nr_curation.yaml`)
   - Data curation pipeline

2. Ask the user for key parameters:
   - `max_concurrent_tasks`: how many tasks to run in parallel (default varies by config)
   - `dataset.cut_off`: how many data points to process (useful for testing with small numbers)
   - `output.path`: where to write results
   - Any resource overrides (e.g., which deployed model to use)

3. Build and run the command:

**Omni Debate example:**
```bash
python -m matrix.agents.p2p_agents --config-name=omni_debate.yaml \
    max_concurrent_tasks=1 \
    dataset.cut_off=3
```

**CoRAL example:**
```bash
python -m matrix.agents.p2p_agents --config-name=coral_mmlu_pro.yaml \
    max_concurrent_tasks=10 \
    dataset.cut_off=10 \
    resources.student_llm.matrix_service=gpt120b \
    resources.teacher_llm.matrix_service=gpt120b \
    resources.extractor_llm.matrix_service=8B \
    output.path=$HOME/temp/coral_test.jsonl
```

**Tau2-Bench example:**
```bash
# Build container first
apptainer build $HOME/temp/tau2_bench.sif matrix/agents/config/containers/tau2_bench.def

python -m matrix.agents.p2p_agents --config-name=tau2_bench.yaml \
    max_concurrent_tasks=10 \
    dataset.cut_off=10 \
    resources.user_simulator_llm.matrix_service=gpt120b \
    resources.llm_agent_llm.matrix_service=gpt120b \
    output.path=$HOME/temp/tau2_test.jsonl \
    tmp_dir=$HOME/temp \
    resources.container.start_config.image=$HOME/temp/tau2_bench.sif \
    domain=telecom
```

4. Monitor the run - output will be written to the specified `output.path` as JSONL.

## Overriding Resources

Resource configs are in `matrix/agents/config/resources/`. Override which deployed model an agent uses:
```bash
resources.<resource_name>.matrix_service=<app_name>
```

Override sampling parameters:
```bash
resources.<resource_name>.sampling_params.max_tokens=10240
resources.<resource_name>.sampling_params.temperature=0.7
```

## Creating Custom Agent Configs

Agent configs use Hydra and live in `matrix/agents/config/`. Each config composes:
- `agents/` - agent definitions
- `dataset/` - data source
- `metrics/` - evaluation metrics
- `orchestrator/` - workflow orchestration
- `resources/` - LLM and service resources

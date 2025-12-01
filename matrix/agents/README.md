# Matrix Peer-to-Peer Data Synthesis

A Python package for multi-agent synthetic data generation. Technical details are in the [paper](https://arxiv.org/abs/2511.21686):
- Peer-to-peer agents running as Ray actors.
- Decentralized orchestration through serializable messages.
- Offload computation to distributed services: LLM and containers
- Three parallelism: data, task and agent.
- Row level scheduling to avoid batch synchronization.
- Support opportunitic compute.
- Reduce serialization overhead through message offloading.


## Examples
### Deploy applications
```bash
matrix deploy_applications --action replace --applications "[{'model_name': 'openai/gpt-oss-120b', 'name': 'gpt120b', 'min_replica': 2, 'enable_tools': 'true', 'use_grpc': 'false'}, {'name': 'container', 'app_type': container, 'min_replica': 1, 'max_ongoing_requests':20}, {'model_name': 'meta-llama/Llama-3.1-8B-Instruct', 'use_grpc': 'true', 'min_replica': 1, 'name': '8B'}]"
```

### Collaborative reasoner
```bash
python -m matrix.agents.p2p_agents --config-name=coral_mmlu_pro.yaml max_concurrent_tasks=10 dataset.cut_off=10 resources.student_llm.matrix_service=gpt120b resources.teacher_llm.matrix_service=gpt120b resources.extractor_llm.matrix_service=8B output.path=$HOME/temp/coral_test.jsonl resources.teacher_llm.sampling_params.max_tokens=10240 resources.student_llm.sampling_params.max_tokens=10240
```

### Tau2-bench
```bash
apptainer build $HOME/temp/tau2_bench.sif matrix/agents/config/containers/tau2_bench.def

python -m matrix.agents.p2p_agents --config-name=tau2_bench.yaml max_concurrent_tasks=10 dataset.cut_off=10 resources.user_simulator_llm.matrix_service=gpt120b resources.llm_agent_llm.matrix_service=gpt120b output.path=$HOME/temp/tau2_test.jsonl tmp_dir=$HOME/temp resources.container.start_config.image=$HOME/temp/tau2_bench.sif domain=telecom
```

### SWE-bench

```bash
python -m matrix.agents.scripts.convert_swebench_to_apptainer --output_dir=$HOME/temp/apptainer_sif

python -m matrix.agents.p2p_agents --config-name=mini_swe_agent max_concurrent_tasks=10 dataset.cut_off=10 resources.coder_llm.matrix_service=gpt120b output.path=$HOME/temp/swe_test.jsonl metrics.pred_file=$HOME/temp/swe_test_preds.json dataset.name=verified debug=false resources.coder_llm.sampling_params.max_tokens=10240 resources.container.exec_params.timeout_secs=180 resources.container.docker_cache=$HOME/temp/apptainer_sif
```

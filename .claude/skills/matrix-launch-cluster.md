---
name: matrix-launch-cluster
description: Launch a Ray cluster on Slurm using matrix start_cluster
user-invocable: true
---

# Launch a Matrix Ray Cluster

Help the user launch a Ray cluster on Slurm.

## Steps

1. Ask the user for the following parameters:
   - **Slurm account** (required) - e.g., `smallomnillm`
   - **Slurm QOS** (required) - e.g., `h200_smallomnillm_high`
   - **Number of workers** (default: 1)
   - **GPUs per node** (default: 8, use 0 for API-only models like Azure OpenAI or Gemini)
   - **Whether to enable Grafana dashboard** (default: no)

2. Run the cluster start command:
```bash
matrix start_cluster --add_workers <num_workers> --slurm "{'account': '<account>', 'qos': '<qos>'}"
```

If Grafana is requested:
```bash
matrix start_cluster --add_workers <num_workers> --slurm "{'account': '<account>', 'qos': '<qos>'}" --enable_grafana
```

If custom GPUs per node:
```bash
matrix start_cluster --add_workers <num_workers> --slurm "{'account': '<account>', 'qos': '<qos>', 'gpus_per_node': <n>}"
```

3. Check allocation status:
```bash
squeue --me
```

4. Check cluster readiness:
```bash
matrix status
```

5. Once the cluster is ready, `matrix status` will print port forwarding information. Help the user set up SSH tunneling for the Ray dashboard. The format is:
```bash
ssh -L <dashboard_port>:localhost:<dashboard_port> -L <client_port>:localhost:<client_port> -L <serve_port>:localhost:<serve_port> <worker_hostname> -J <jump_host>
```

The ports come from `matrix status` output. The jump host is cluster-specific (e.g., `fair-sc` for the fair-sc cluster). The user needs to have the jump host configured in their `~/.ssh/config`.

After tunneling, visit the Ray dashboard at `http://localhost:<dashboard_port>` in a browser.

## Adding More Workers Later

```bash
matrix start_cluster --add_workers <additional_workers> --slurm "{'account': '<account>', 'qos': '<qos>'}"
```

## Stopping the Cluster

```bash
matrix stop_cluster
```

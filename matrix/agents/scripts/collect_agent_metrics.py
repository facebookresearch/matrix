# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Collect agent metrics from JSONL output files.

Parses completion tokens for agents, counts total rows,
lost rows (dead orchestrator), and error rows.

Usage:
    python collect_agent_metrics.py sample.jsonl
    python collect_agent_metrics.py sample.jsonl --agents teacher student
    python collect_agent_metrics.py sample.jsonl --json
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import fire


def load_jsonl(filepath: str) -> list[dict[str, Any]]:
    """Load JSONL file (supports .jsonl and .jsonl.zst) and return list of records."""
    records = []
    path = Path(filepath)

    if path.suffix == ".zst" or filepath.endswith(".jsonl.zst"):
        import zstandard as zstd

        with open(filepath, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                import io

                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text_stream:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
    else:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    return records


def collect_agent_metrics(
    records: list[dict[str, Any]],
    agents: list[str] | None = None,
) -> dict[str, Any]:
    """
    Collect metrics from agent records.

    Args:
        records: List of parsed JSONL records
        agents: Optional list of agent names to filter. If None, collect all agents.

    Returns dict with:
        - total_rows: Total number of records
        - lost_rows: Number of records with status="lost" (dead orchestrator)
        - error_rows: Number of records where history is non-empty and last
                      entry has status_ok=False
        - agent_completion_tokens: Dict mapping agent name -> total completion tokens
        - agent_prompt_tokens: Dict mapping agent name -> total prompt tokens
        - agent_total_tokens: Dict mapping agent name -> total tokens
        - agent_call_counts: Dict mapping agent name -> number of LLM calls
    """
    agents_set = set(agents) if agents else None
    total_rows = 0
    lost_rows = 0
    error_rows = 0
    error_examples: list[dict[str, Any]] = []

    agent_completion_tokens: dict[str, int] = defaultdict(int)
    agent_prompt_tokens: dict[str, int] = defaultdict(int)
    agent_total_tokens: dict[str, int] = defaultdict(int)
    agent_call_counts: dict[str, int] = defaultdict(int)

    for record in records:
        total_rows += 1

        # Check for lost status (dead orchestrator)
        status = record.get("status")
        if status == "lost":
            lost_rows += 1
            continue  # Lost rows don't have history

        # Process history for token counts and error detection
        history = record.get("history", [])

        # Check for error: history is non-empty and last entry has status_ok=False
        if history:
            last_entry = history[-1]
            last_response = last_entry.get("response", {})
            if last_response.get("status_ok") is False:
                error_rows += 1
                if len(error_examples) < 3:
                    error_examples.append(
                        {
                            "id": record.get("id"),
                            "agent": last_entry.get("agent"),
                            "response": last_response,
                        }
                    )

        # Collect token usage from all history entries
        for entry in history:
            agent = entry.get("agent", "unknown")

            # Skip if filtering by agents and this agent is not in the list
            if agents_set is not None and agent not in agents_set:
                continue

            response = entry.get("response", {})
            usage = response.get("usage", {})

            if usage:
                completion_tokens = usage.get("completion_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)

                agent_completion_tokens[agent] += completion_tokens
                agent_prompt_tokens[agent] += prompt_tokens
                agent_total_tokens[agent] += total_tokens
                agent_call_counts[agent] += 1

    return {
        "total_rows": total_rows,
        "lost_rows": lost_rows,
        "error_rows": error_rows,
        "error_examples": error_examples,
        "agent_completion_tokens": dict(agent_completion_tokens),
        "agent_prompt_tokens": dict(agent_prompt_tokens),
        "agent_total_tokens": dict(agent_total_tokens),
        "agent_call_counts": dict(agent_call_counts),
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    """Print metrics in a formatted way."""
    print("\n" + "=" * 60)
    print(" AGENT METRICS SUMMARY")
    print("=" * 60)

    print(f"\nTotal rows:  {metrics['total_rows']}")
    print(f"Lost rows:   {metrics['lost_rows']} (dead orchestrator)")
    print(
        f"Error rows:  {metrics['error_rows']} (last history entry has status_ok=False)"
    )

    # Calculate percentages
    total = metrics["total_rows"]
    if total > 0:
        lost_pct = metrics["lost_rows"] / total * 100
        error_pct = metrics["error_rows"] / total * 100
        success_rows = total - metrics["lost_rows"] - metrics["error_rows"]
        success_pct = success_rows / total * 100
        print(f"\nSuccess rate: {success_pct:.2f}% ({success_rows}/{total})")
        print(f"Lost rate:    {lost_pct:.2f}%")
        print(f"Error rate:   {error_pct:.2f}%")

    # Print token usage per agent
    print("\n" + "-" * 60)
    print(" TOKEN USAGE BY AGENT")
    print("-" * 60)

    agents = sorted(set(metrics["agent_completion_tokens"].keys()))
    if agents:
        # Header
        print(
            f"{'Agent':<20} {'Calls':>10} {'Prompt':>12} {'Completion':>12} {'Total':>12}"
        )
        print("-" * 60)

        for agent in agents:
            calls = metrics["agent_call_counts"].get(agent, 0)
            prompt = metrics["agent_prompt_tokens"].get(agent, 0)
            completion = metrics["agent_completion_tokens"].get(agent, 0)
            total = metrics["agent_total_tokens"].get(agent, 0)
            print(f"{agent:<20} {calls:>10} {prompt:>12} {completion:>12} {total:>12}")

        # Totals
        print("-" * 60)
        total_calls = sum(metrics["agent_call_counts"].values())
        total_prompt = sum(metrics["agent_prompt_tokens"].values())
        total_completion = sum(metrics["agent_completion_tokens"].values())
        total_total = sum(metrics["agent_total_tokens"].values())
        print(
            f"{'TOTAL':<20} {total_calls:>10} {total_prompt:>12} {total_completion:>12} {total_total:>12}"
        )
    else:
        print("No token usage data found.")

    # Print error examples if any
    if metrics["error_examples"]:
        print("\n" + "-" * 60)
        print(" ERROR EXAMPLES (first 3)")
        print("-" * 60)
        for i, ex in enumerate(metrics["error_examples"], 1):
            print(f"\n[{i}] ID: {ex['id']}, Agent: {ex['agent']}")
            resp = ex["response"]
            # Print relevant fields
            for key in ["status_ok", "error", "text"]:
                if key in resp:
                    val = resp[key]
                    if isinstance(val, str) and len(val) > 200:
                        val = val[:200] + "..."
                    print(f"    {key}: {val}")

    print()


def main(
    input_file: str,
    agents: list[str] | None = None,
    json_output: bool = False,
) -> None:
    """
    Collect agent metrics from JSONL output files.

    Args:
        input_file: Path to input JSONL file (supports .jsonl and .jsonl.zst)
        agents: List of agent names to collect metrics for (e.g., teacher student).
                If not specified, collect all agents.
        json_output: Output metrics as JSON instead of formatted text.
    """
    print(f"Loading data from {input_file}...")
    records = load_jsonl(input_file)
    print(f"Loaded {len(records)} records")

    if agents:
        print(f"Filtering for agents: {', '.join(agents)}")

    metrics = collect_agent_metrics(records, agents=agents)

    if json_output:
        # Remove error_examples for cleaner JSON output
        output = {k: v for k, v in metrics.items() if k != "error_examples"}
        print(json.dumps(output, indent=2))
    else:
        print_metrics(metrics)


if __name__ == "__main__":
    fire.Fire(main)

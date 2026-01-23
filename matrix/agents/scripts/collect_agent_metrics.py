# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Collect and analyze agent metrics from instrumentation data.

Computes end-to-end time/throughput breakdown:
- Per-task latency contributions (dequeue, handle per role)
- Network bandwidth analysis with sliding window
"""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
                    records.append(json.loads(line))
    else:
        with open(filepath, "r") as f:
            for line in f:
                records.append(json.loads(line))

    return records


def load_jsonl_glob(pattern: str) -> list[dict[str, Any]]:
    """
    Load records from files matching a glob pattern.

    Supports both .jsonl and .jsonl.zst files.

    Args:
        pattern: Glob pattern (e.g., "results/*.jsonl", "data/**/*.jsonl.zst")

    Returns:
        Combined list of records from all matching files.
    """
    # Expand glob pattern
    files = sorted(glob.glob(pattern, recursive=True))

    # Filter to only jsonl and jsonl.zst files
    valid_files = [f for f in files if f.endswith(".jsonl") or f.endswith(".jsonl.zst")]

    if not valid_files:
        raise FileNotFoundError(
            f"No .jsonl or .jsonl.zst files found matching: {pattern}"
        )

    all_records = []
    for filepath in valid_files:
        print(f"  Loading {filepath}...")
        records = load_jsonl(filepath)
        print(f"    -> {len(records)} records")
        all_records.extend(records)

    return all_records


def discover_latency_metrics(records: list[dict[str, Any]]) -> dict[str, set[str]]:
    """
    Scan records to discover all latency metric types and their roles.

    Returns dict mapping metric_type -> set of roles that have that metric.
    Only includes metrics ending in '_seconds'.
    """
    metrics_to_roles = defaultdict(set)

    for record in records:
        instrumentation = record.get("instrumentation", [])
        for item in instrumentation:
            metric_type = item[0]
            role = item[1]
            # Only latency metrics (ending in _seconds)
            if metric_type.endswith("_seconds"):
                metrics_to_roles[metric_type].add(role)

    return dict(metrics_to_roles)


def analyze_latency_breakdown(records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Analyze per-task latency breakdown.

    Dynamically discovers all latency metrics from the data and computes:
    - e2e_delay: finish_timestamp - creation_timestamp
    - init_seconds: init_timestamp - creation_timestamp
    - For each discovered metric_type ending in '_seconds':
      - sum_{metric_type}: sum across all roles
      - {metric_type}_{role}: value per role (for metrics with multiple roles)
    - percentage contributions to e2e delay for each
    """
    # First pass: discover all latency metrics and roles
    metrics_to_roles = discover_latency_metrics(records)

    task_metrics = []

    for record in records:
        task_id = record.get("id", None)
        creation_ts = record.get("creation_timestamp", 0)
        init_ts = record.get("init_timestamp", 0)
        finish_ts = record.get("finish_timestamp", 0)
        e2e_delay = finish_ts - creation_ts
        init_seconds = init_ts - creation_ts

        instrumentation = record.get("instrumentation", [])

        # Aggregate latencies: (metric_type, role) -> sum of values
        latency_by_metric_role = defaultdict(float)

        for item in instrumentation:
            metric_type = item[0]
            role = item[1]
            value = item[2]

            if metric_type.endswith("_seconds"):
                latency_by_metric_role[(metric_type, role)] += value

        # Build task data
        task_data = {
            "task_id": task_id,
            "e2e_delay_seconds": e2e_delay,
            "init_seconds": init_seconds,
            "pct_init": (init_seconds / e2e_delay * 100) if e2e_delay > 0 else 0,
        }

        # For each metric type, add sum and per-role breakdown
        for metric_type, roles in metrics_to_roles.items():
            # Calculate sum across all roles for this metric
            metric_sum = sum(
                latency_by_metric_role.get((metric_type, role), 0) for role in roles
            )
            # Remove "_seconds" suffix for cleaner naming
            metric_base = metric_type.replace("_seconds", "")
            task_data[f"sum_{metric_base}_seconds"] = metric_sum
            task_data[f"pct_sum_{metric_base}"] = (
                (metric_sum / e2e_delay * 100) if e2e_delay > 0 else 0
            )

            # If multiple roles, also add per-role breakdown
            if len(roles) > 1:
                for role in roles:
                    role_val = latency_by_metric_role.get((metric_type, role), 0)
                    task_data[f"{metric_base}_{role}_seconds"] = role_val
                    task_data[f"pct_{metric_base}_{role}"] = (
                        (role_val / e2e_delay * 100) if e2e_delay > 0 else 0
                    )

        task_metrics.append(task_data)

    return pd.DataFrame(task_metrics)


def compute_latency_summary_stats(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute summary statistics for latency metrics.

    Returns two DataFrames:
    - absolute_stats: stats for absolute time columns (ending in _seconds)
    - percent_stats: stats for percentage columns (starting with pct_)
    Both sorted by median in descending order.
    """
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "task_id" in numeric_cols:
        numeric_cols.remove("task_id")

    # Separate absolute and percentage columns
    abs_cols = [
        c for c in numeric_cols if c.endswith("_seconds") or c == "e2e_delay_seconds"
    ]
    pct_cols = [c for c in numeric_cols if c.startswith("pct_")]

    def compute_stats(cols):
        if not cols:
            return pd.DataFrame()
        stats = df[cols].agg(["count", "mean", "std", "min", "median", "max"])
        stats.loc["p50"] = df[cols].quantile(0.5)
        stats.loc["p90"] = df[cols].quantile(0.9)
        stats.loc["p99"] = df[cols].quantile(0.99)
        stats_t = stats.T
        # Sort by median descending
        stats_t = stats_t.sort_values("median", ascending=False)
        return stats_t

    return compute_stats(abs_cols), compute_stats(pct_cols)


def extract_events_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Extract all ser_size_kb events from records.

    Handles both old (ser_seize_kb) and new (ser_size_kb) metric names.

    Returns DataFrame with columns: task_id, role, timestamp, kb
    """
    events = []

    for record in records:
        instrumentation = record.get("instrumentation", [])
        task_id = record.get("id", None)

        for item in instrumentation:
            metric_type = item[0]
            role = item[1]
            value = item[2]

            # Handle both old (ser_seize_kb) and new (ser_size_kb) names
            if metric_type in ("ser_size_kb", "ser_seize_kb"):
                timestamp, kb = value
                events.append(
                    {
                        "task_id": task_id,
                        "role": role,
                        "timestamp": timestamp,
                        "kb": kb,
                    }
                )

    if not events:
        return pd.DataFrame()

    df = pd.DataFrame(events)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def analyze_throughput_breakdown(
    df_events: pd.DataFrame, window_seconds: float = 60.0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Analyze per-role throughput (events per second) with sliding window.

    For each event, calculate the throughput of each role in the last window_seconds.
    This helps identify which agent is the bottleneck.

    Args:
        df_events: DataFrame with columns: task_id, role, timestamp, kb
        window_seconds: Sliding window size (default 60s)

    Returns:
        - DataFrame with per-event throughput by role
        - DataFrame with per-role throughput statistics
        - Series with overall throughput statistics
    """
    if df_events.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series()

    timestamps = df_events["timestamp"].values
    roles = df_events["role"].values
    unique_roles = df_events["role"].unique()
    n = len(timestamps)
    time_span = timestamps.max() - timestamps.min()

    # Use binary search for efficient window lookups
    window_starts = timestamps - window_seconds
    # For each event, find the first event index in window using searchsorted
    left_indices = np.searchsorted(timestamps, window_starts, side="left")

    # Pre-compute cumulative counts per role for O(1) range queries
    role_cumsum = {}
    for role in unique_roles:
        role_mask = (roles == role).astype(np.int32)
        role_cumsum[role] = np.concatenate([[0], np.cumsum(role_mask)])

    # Total cumsum
    total_cumsum = np.concatenate([[0], np.arange(1, n + 1)])

    # Vectorized computation of counts in window
    throughput_data = {
        "timestamp": timestamps,
        "role": roles,
        "task_id": df_events["task_id"].values,
    }

    # Window durations: min(window_seconds, ts - first_event_in_window)
    first_in_window_ts = timestamps[left_indices]
    window_durations = np.minimum(window_seconds, timestamps - first_in_window_ts)
    window_durations = np.maximum(window_durations, 1e-6)  # Avoid division by zero

    # Compute per-role throughput
    for role in unique_roles:
        cs = role_cumsum[role]
        # Count in window = cumsum[i+1] - cumsum[left_idx]
        counts = cs[np.arange(n) + 1] - cs[left_indices]
        throughput_data[f"throughput_{role}_per_sec"] = counts / window_durations
        throughput_data[f"count_{role}_in_window"] = counts

    # Total throughput
    total_counts = (np.arange(n) + 1) - left_indices
    throughput_data["throughput_total_per_sec"] = total_counts / window_durations
    throughput_data["count_total_in_window"] = total_counts

    df_throughput = pd.DataFrame(throughput_data)

    # Calculate per-role statistics
    role_stats_data = []
    for role in unique_roles:
        col = f"throughput_{role}_per_sec"
        role_event_count = (roles == role).sum()
        vals = df_throughput[col]
        role_stats_data.append(
            {
                "role": role,
                "total_events": role_event_count,
                "avg_throughput_events_per_sec": (
                    role_event_count / time_span if time_span > 0 else 0
                ),
                "throughput_mean": vals.mean(),
                "throughput_std": vals.std(),
                "throughput_min": vals.min(),
                "throughput_p50": vals.quantile(0.5),
                "throughput_p90": vals.quantile(0.9),
                "throughput_max": vals.max(),
            }
        )

    df_role_stats = pd.DataFrame(role_stats_data).set_index("role")

    # Overall statistics
    overall_stats = {
        "total_events": len(df_events),
        "time_span_seconds": time_span,
        "avg_throughput_events_per_sec": (
            len(df_events) / time_span if time_span > 0 else 0
        ),
        "throughput_total_mean": df_throughput["throughput_total_per_sec"].mean(),
        "throughput_total_std": df_throughput["throughput_total_per_sec"].std(),
        "throughput_total_p50": df_throughput["throughput_total_per_sec"].quantile(0.5),
        "throughput_total_p90": df_throughput["throughput_total_per_sec"].quantile(0.9),
        "throughput_total_max": df_throughput["throughput_total_per_sec"].max(),
    }

    return df_throughput, df_role_stats, pd.Series(overall_stats)


def analyze_network_bandwidth(
    df_events: pd.DataFrame, window_seconds: float = 60.0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Analyze network bandwidth from ser_size_kb events.

    Args:
        df_events: DataFrame with columns: task_id, role, timestamp, kb
        window_seconds: Sliding window size for bandwidth calculation (default 60s)

    Returns:
        - DataFrame of all bandwidth events with bandwidth columns
        - DataFrame of per-role bandwidth statistics
        - Series with overall bandwidth statistics
    """
    if df_events.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series()

    df = df_events.copy()
    timestamps = df["timestamp"].values
    kb_values = df["kb"].values
    roles = df["role"].values
    unique_roles = df["role"].unique()
    n = len(timestamps)

    # Calculate total KB and time span
    total_kb = kb_values.sum()
    time_span = timestamps.max() - timestamps.min()

    # Use binary search for efficient window lookups
    window_starts = timestamps - window_seconds
    left_indices = np.searchsorted(timestamps, window_starts, side="left")

    # Pre-compute cumulative KB sums per role for O(1) range queries
    role_kb_cumsum = {}
    for role in unique_roles:
        role_mask = roles == role
        role_kb = np.where(role_mask, kb_values, 0)
        role_kb_cumsum[role] = np.concatenate([[0], np.cumsum(role_kb)])

    # Total KB cumsum
    total_kb_cumsum = np.concatenate([[0], np.cumsum(kb_values)])

    # Window durations
    first_in_window_ts = timestamps[left_indices]
    window_durations = np.minimum(window_seconds, timestamps - first_in_window_ts)
    window_durations = np.maximum(window_durations, 1e-6)  # Avoid division by zero

    # Compute total bandwidth (all roles)
    indices = np.arange(n)
    total_kb_in_window = total_kb_cumsum[indices + 1] - total_kb_cumsum[left_indices]
    df["bandwidth_total_kb_per_sec"] = total_kb_in_window / window_durations

    # Compute per-role bandwidth
    for role in unique_roles:
        cs = role_kb_cumsum[role]
        kb_in_window = cs[indices + 1] - cs[left_indices]
        df[f"bandwidth_{role}_kb_per_sec"] = kb_in_window / window_durations

    # Calculate per-role statistics
    role_stats_data = []
    for role in unique_roles:
        role_mask = roles == role
        role_bw_col = f"bandwidth_{role}_kb_per_sec"
        role_kb = kb_values[role_mask]

        role_stats_data.append(
            {
                "role": role,
                "event_count": role_mask.sum(),
                "total_kb": role_kb.sum(),
                "kb_mean": role_kb.mean() if len(role_kb) > 0 else 0,
                "kb_std": role_kb.std() if len(role_kb) > 0 else 0,
                "avg_bandwidth_kb_per_sec": (
                    role_kb.sum() / time_span if time_span > 0 else 0
                ),
                "bandwidth_mean": df[role_bw_col].mean(),
                "bandwidth_std": df[role_bw_col].std(),
                "bandwidth_p50": df[role_bw_col].quantile(0.5),
                "bandwidth_p90": df[role_bw_col].quantile(0.9),
                "bandwidth_max": df[role_bw_col].max(),
            }
        )

    df_role_stats = pd.DataFrame(role_stats_data).set_index("role")

    # Overall statistics
    overall_stats = {
        "total_kb": total_kb,
        "total_events": len(df),
        "time_span_seconds": time_span,
        "avg_bandwidth_kb_per_sec": total_kb / time_span if time_span > 0 else 0,
        "bandwidth_total_mean": df["bandwidth_total_kb_per_sec"].mean(),
        "bandwidth_total_std": df["bandwidth_total_kb_per_sec"].std(),
        "bandwidth_total_p50": df["bandwidth_total_kb_per_sec"].quantile(0.5),
        "bandwidth_total_p90": df["bandwidth_total_kb_per_sec"].quantile(0.9),
        "bandwidth_total_p99": df["bandwidth_total_kb_per_sec"].quantile(0.99),
        "bandwidth_total_max": df["bandwidth_total_kb_per_sec"].max(),
    }

    return df, df_role_stats, pd.Series(overall_stats)


def print_table(title: str, df: pd.DataFrame) -> None:
    """Print a formatted table with title."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)
    print(df.to_string())
    print()


def print_bandwidth_timeseries(
    df: pd.DataFrame,
    bw_col: str = "bandwidth_total_kb_per_sec",
    width: int = 60,
    num_bins: int = 50,
    spike_threshold_mult: float = 100.0,
    extreme_threshold_mult: float = 1000.0,
    early_focus_ratio: float = 0.5,
) -> None:
    """
    Print a text-based time series visualization of bandwidth.

    Args:
        df: DataFrame with timestamp and bandwidth columns
        bw_col: Column name for bandwidth values
        width: Width of the bar chart
        num_bins: Number of time bins to aggregate into
        spike_threshold_mult: Mark spikes where bandwidth > p90 * this multiplier
        extreme_threshold_mult: Threshold for extreme bandwidth analysis (default 1000x p90)
        early_focus_ratio: Fraction of bins to allocate to first 10% of time
    """
    if df.empty or bw_col not in df.columns:
        print("No bandwidth data to visualize.")
        return

    timestamps = df["timestamp"].values
    bw_values = df[bw_col].values

    t_min, t_max = timestamps.min(), timestamps.max()
    t_range = t_max - t_min
    bw_p90 = np.percentile(bw_values, 90)
    bw_max = bw_values.max()
    spike_threshold = bw_p90 * spike_threshold_mult

    print("\n" + "=" * 80)
    print(f" BANDWIDTH TIME SERIES (non-uniform bins, more detail at start)")
    print("=" * 80)
    print(f"Time range: {t_range:.2f}s, p90={bw_p90:.2f} KB/s, max={bw_max:.2f} KB/s")
    print(
        f"Spike threshold: > {spike_threshold:.2f} KB/s (>{spike_threshold_mult}x p90)"
    )
    print(f"Legend: [=] normal, [#] > p90, [!] > {spike_threshold_mult}x p90 (spike)")
    print("-" * 80)

    # Create non-uniform bins: more bins at the start
    # First 10% of time gets early_focus_ratio of bins
    early_time = 0.1 * t_range
    early_bins = int(num_bins * early_focus_ratio)
    late_bins = num_bins - early_bins

    if early_bins > 0 and late_bins > 0:
        early_edges = np.linspace(t_min, t_min + early_time, early_bins + 1)
        late_edges = np.linspace(t_min + early_time, t_max, late_bins + 1)[
            1:
        ]  # Skip first to avoid duplicate
        bin_edges = np.concatenate([early_edges, late_edges])
    else:
        bin_edges = np.linspace(t_min, t_max, num_bins + 1)

    actual_bins = len(bin_edges) - 1
    bin_max_bw = np.zeros(actual_bins)
    bin_has_spike = np.zeros(actual_bins, dtype=bool)
    bin_spike_times = [[] for _ in range(actual_bins)]

    for i in range(actual_bins):
        mask = (timestamps >= bin_edges[i]) & (timestamps < bin_edges[i + 1])
        if mask.any():
            bin_bw = bw_values[mask]
            bin_max_bw[i] = bin_bw.max()
            # Track spikes
            spike_mask = bin_bw > spike_threshold
            if spike_mask.any():
                bin_has_spike[i] = True
                spike_ts = timestamps[mask][spike_mask]
                spike_bw = bin_bw[spike_mask]
                for t, b in zip(spike_ts, spike_bw):
                    bin_spike_times[i].append((t, b))

    # Normalize for display (use log scale if range is large)
    display_max = max(bin_max_bw.max(), 1)
    use_log = (display_max / max(bw_p90, 1)) > 100

    if use_log:
        print(f"Using log scale due to large range")
        log_values = np.log10(bin_max_bw + 1)
        log_max = np.log10(display_max + 1)
        normalized = log_values / log_max
    else:
        normalized = bin_max_bw / display_max

    # Print the time series
    print(f"\n[EARLY PHASE: first {early_time:.2f}s with {early_bins} bins]")
    for i in range(actual_bins):
        t_start = bin_edges[i] - t_min
        t_end = bin_edges[i + 1] - t_min
        bar_len = int(normalized[i] * width)

        # Choose character based on value
        if bin_has_spike[i]:
            char = "!"
        elif bin_max_bw[i] > bw_p90:
            char = "#"
        else:
            char = "="

        bar = char * bar_len
        marker = " <<<SPIKE" if bin_has_spike[i] else ""

        # Add phase separator
        if i == early_bins:
            print(
                f"\n[LATE PHASE: remaining {t_range - early_time:.2f}s with {late_bins} bins]"
            )

        print(
            f"[{t_start:7.3f}s-{t_end:7.3f}s] |{bar:<{width}}| {bin_max_bw[i]:10.2f} KB/s{marker}"
        )

    # Print spike details
    spike_events = []
    for i in range(actual_bins):
        for t, b in bin_spike_times[i]:
            spike_events.append((t, b))

    if spike_events:
        print("\n" + "-" * 80)
        print(
            f"SPIKE DETAILS ({len(spike_events)} events > {spike_threshold:.2f} KB/s):"
        )
        print("-" * 80)
        # Sort by bandwidth descending
        spike_events.sort(key=lambda x: -x[1])
        for t, b in spike_events[:20]:  # Show top 20
            relative_t = t - t_min
            mult = b / bw_p90 if bw_p90 > 0 else 0
            print(f"  t={relative_t:8.4f}s: {b:12.2f} KB/s ({mult:6.1f}x p90)")
        if len(spike_events) > 20:
            print(f"  ... and {len(spike_events) - 20} more spikes")

    # Calculate total time when bandwidth is extremely high (>extreme_threshold_mult x p90)
    extreme_threshold = bw_p90 * extreme_threshold_mult
    extreme_mask = bw_values > extreme_threshold
    extreme_count = extreme_mask.sum()

    if extreme_count > 0:
        # Estimate duration: for each extreme event, estimate its "duration"
        # as time to next event or average inter-event time
        extreme_timestamps = timestamps[extreme_mask]
        if len(timestamps) > 1:
            avg_inter_event = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
        else:
            avg_inter_event = 0.001  # default 1ms

        # Sum up durations for extreme events
        extreme_durations = []
        for i, t in enumerate(extreme_timestamps):
            # Find next timestamp
            idx = np.searchsorted(timestamps, t)
            if idx < len(timestamps) - 1:
                duration = timestamps[idx + 1] - t
            else:
                duration = avg_inter_event
            extreme_durations.append(duration)

        total_extreme_time = sum(extreme_durations)
        pct_extreme = (total_extreme_time / t_range * 100) if t_range > 0 else 0

        print("\n" + "-" * 80)
        print(
            f"EXTREME BANDWIDTH (> {extreme_threshold_mult:.0f}x p90 = {extreme_threshold:.2f} KB/s):"
        )
        print("-" * 80)
        print(
            f"  Events with bandwidth > {extreme_threshold_mult:.0f}x p90: {extreme_count}"
        )
        print(
            f"  Estimated total duration: {total_extreme_time:.4f}s ({pct_extreme:.2f}% of total time)"
        )
        print(
            f"  Time range affected: {extreme_timestamps.min() - t_min:.4f}s to {extreme_timestamps.max() - t_min:.4f}s"
        )

        # Show top extreme events
        extreme_events = [
            (t, b) for t, b in zip(timestamps[extreme_mask], bw_values[extreme_mask])
        ]
        extreme_events.sort(key=lambda x: -x[1])
        print(f"  Top 10 extreme events:")
        for t, b in extreme_events[:10]:
            relative_t = t - t_min
            mult = b / bw_p90 if bw_p90 > 0 else 0
            print(f"    t={relative_t:8.4f}s: {b:12.2f} KB/s ({mult:6.1f}x p90)")
    else:
        print(
            f"\nNo events with bandwidth > {extreme_threshold_mult:.0f}x p90 ({extreme_threshold:.2f} KB/s)"
        )

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Collect and analyze agent metrics from instrumentation data"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input JSONL file or glob pattern (e.g., 'results/*.jsonl', 'data/**/*.jsonl.zst')",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=60.0,
        help="Sliding window size in seconds for bandwidth/throughput calculation (default: 60)",
    )
    parser.add_argument(
        "--extreme-mult",
        type=float,
        default=1000.0,
        help="Multiplier for extreme bandwidth threshold (threshold = p90 * mult, default: 1000)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional prefix for output CSV files",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.input_file}...")

    # Check if input is a glob pattern or single file
    if "*" in args.input_file or "?" in args.input_file:
        records = load_jsonl_glob(args.input_file)
    else:
        records = load_jsonl(args.input_file)

    print(f"Loaded {len(records)} task records total")

    # Discover latency metrics
    metrics_to_roles = discover_latency_metrics(records)
    print("\nDiscovered latency metrics:")
    for metric_type, roles in sorted(metrics_to_roles.items()):
        print(f"  - {metric_type}: roles={sorted(roles)}")

    # Analyze latency breakdown
    print("\nAnalyzing latency breakdown...")
    df_latency = analyze_latency_breakdown(records)
    latency_abs_stats, latency_pct_stats = compute_latency_summary_stats(df_latency)

    print_table(
        "LATENCY BREAKDOWN - ALL TASKS (absolute, sorted by median desc)",
        latency_abs_stats,
    )
    print_table(
        "LATENCY BREAKDOWN - ALL TASKS (percentages, sorted by median desc)",
        latency_pct_stats,
    )

    # Analyze tasks by latency percentiles
    e2e_p25 = df_latency["e2e_delay_seconds"].quantile(0.25)
    e2e_p50 = df_latency["e2e_delay_seconds"].quantile(0.5)
    e2e_p90 = df_latency["e2e_delay_seconds"].quantile(0.9)

    df_below_p25 = df_latency[df_latency["e2e_delay_seconds"] < e2e_p25]
    df_above_p50 = df_latency[df_latency["e2e_delay_seconds"] >= e2e_p50]
    df_above_p90 = df_latency[df_latency["e2e_delay_seconds"] >= e2e_p90]

    print(
        f"\ne2e_delay thresholds: p25={e2e_p25:.4f}s, p50={e2e_p50:.4f}s, p90={e2e_p90:.4f}s"
    )
    print(
        f"Tasks below p25: {len(df_below_p25)}, above p50: {len(df_above_p50)}, above p90: {len(df_above_p90)}"
    )

    # Below p25 (fast tasks)
    abs_below_p25, pct_below_p25 = compute_latency_summary_stats(df_below_p25)
    print_table(
        f"LATENCY - TASKS BELOW P25 (< {e2e_p25:.4f}s, n={len(df_below_p25)}) absolute",
        abs_below_p25,
    )
    print_table(
        f"LATENCY - TASKS BELOW P25 (< {e2e_p25:.4f}s, n={len(df_below_p25)}) percentages",
        pct_below_p25,
    )

    # Above p50 (slow tasks)
    abs_above_p50, pct_above_p50 = compute_latency_summary_stats(df_above_p50)
    print_table(
        f"LATENCY - TASKS ABOVE P50 (>= {e2e_p50:.4f}s, n={len(df_above_p50)}) absolute",
        abs_above_p50,
    )
    print_table(
        f"LATENCY - TASKS ABOVE P50 (>= {e2e_p50:.4f}s, n={len(df_above_p50)}) percentages",
        pct_above_p50,
    )

    # Above p90 (very slow tasks)
    abs_above_p90, pct_above_p90 = compute_latency_summary_stats(df_above_p90)
    print_table(
        f"LATENCY - TASKS ABOVE P90 (>= {e2e_p90:.4f}s, n={len(df_above_p90)}) absolute",
        abs_above_p90,
    )
    print_table(
        f"LATENCY - TASKS ABOVE P90 (>= {e2e_p90:.4f}s, n={len(df_above_p90)}) percentages",
        pct_above_p90,
    )

    # Show per-task sample
    print_table(
        "LATENCY BREAKDOWN - SAMPLE (first 10 tasks)",
        df_latency.head(10),
    )

    # Extract events for bandwidth and throughput analysis
    print("\nExtracting events from records...")
    df_events = extract_events_from_records(records)

    if not df_events.empty:
        print(f"Extracted {len(df_events)} events")

        # Analyze throughput breakdown (events per second per role)
        print("\nAnalyzing throughput breakdown (per-role event rate)...")
        df_throughput, throughput_role_stats, overall_tp_stats = (
            analyze_throughput_breakdown(df_events, window_seconds=args.window)
        )

        print_table(
            "THROUGHPUT BREAKDOWN - OVERALL STATISTICS",
            overall_tp_stats.to_frame("value"),
        )
        print_table(
            "THROUGHPUT BREAKDOWN - PER-ROLE STATISTICS (events/sec)",
            throughput_role_stats,
        )

        # Show throughput columns only for sample
        tp_sample_cols = [
            "timestamp",
            "role",
            "task_id",
            "throughput_total_per_sec",
        ] + [
            c
            for c in df_throughput.columns
            if c.startswith("throughput_") and c != "throughput_total_per_sec"
        ]
        print_table(
            "THROUGHPUT BREAKDOWN - SAMPLE EVENTS (first 20)",
            df_throughput[tp_sample_cols].head(20),
        )

        # Analyze network bandwidth
        print("\nAnalyzing network bandwidth...")
        df_bandwidth, bw_role_stats, overall_bw_stats = analyze_network_bandwidth(
            df_events, window_seconds=args.window
        )

        print_table(
            "NETWORK BANDWIDTH - OVERALL STATISTICS", overall_bw_stats.to_frame("value")
        )
        print_table("NETWORK BANDWIDTH - PER-ROLE STATISTICS", bw_role_stats)
        print_table(
            "NETWORK BANDWIDTH - SAMPLE EVENTS (first 20)",
            df_bandwidth.head(20),
        )

        # Print bandwidth time series visualization
        print_bandwidth_timeseries(
            df_bandwidth,
            spike_threshold_mult=100.0,
            extreme_threshold_mult=args.extreme_mult,
        )

        # Save to CSV if output prefix specified
        if args.output_prefix:
            df_latency.to_csv(f"{args.output_prefix}_latency_per_task.csv", index=False)
            latency_abs_stats.to_csv(f"{args.output_prefix}_latency_abs_stats.csv")
            latency_pct_stats.to_csv(f"{args.output_prefix}_latency_pct_stats.csv")
            df_throughput.to_csv(
                f"{args.output_prefix}_throughput_events.csv", index=False
            )
            throughput_role_stats.to_csv(
                f"{args.output_prefix}_throughput_role_stats.csv"
            )
            overall_tp_stats.to_frame("value").to_csv(
                f"{args.output_prefix}_throughput_overall_stats.csv"
            )
            df_bandwidth.to_csv(
                f"{args.output_prefix}_bandwidth_events.csv", index=False
            )
            bw_role_stats.to_csv(f"{args.output_prefix}_bandwidth_role_stats.csv")
            overall_bw_stats.to_frame("value").to_csv(
                f"{args.output_prefix}_bandwidth_overall_stats.csv"
            )
            print(f"\nResults saved with prefix: {args.output_prefix}")
    else:
        print("No ser_size_kb events found for bandwidth/throughput analysis.")

        # Save latency results only
        if args.output_prefix:
            df_latency.to_csv(f"{args.output_prefix}_latency_per_task.csv", index=False)
            latency_abs_stats.to_csv(f"{args.output_prefix}_latency_abs_stats.csv")
            latency_pct_stats.to_csv(f"{args.output_prefix}_latency_pct_stats.csv")
            print(f"\nLatency results saved with prefix: {args.output_prefix}")


if __name__ == "__main__":
    main()

"""Summarise an online training benchmark from the rank JSONL metrics.

The trainer already writes one ``rank-XXX.jsonl`` per rank under ``<out>/metrics``; this script turns them into the
numbers a tuning decision needs, with the conventions the rest of the repo uses:

* step time is the slowest rank's ``perf/step_ms`` (that is what the cluster waited for);
* samples/s uses the summed ``data/samples`` over ranks and the slowest step;
* MFU uses :func:`compute_cosmos3_flops` (fwd 20.34 TFLOP/sample, x4 with full activation checkpointing) against
  the 910B2 Cube peak of 353.9 TFLOPS/NPU.

    python scripts/summarize_online_metrics.py --metrics-dir /mnt/.../runs/online_base/metrics --world-size 8

Add ``--json`` to emit a machine-readable summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--drop-first", type=int, default=1, help="Warmup steps to exclude from steady state.")
    parser.add_argument("--include-vae-flops", action="store_true", help="Add the 33 TFLOP/sample VAE forward.")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def load_rank(path: Path):
    records = {}
    with path.open() as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "record":
                records[int(payload["step"])] = payload
    return records


def number(record, section, key, default=0.0):
    """Read one leaf from the nested record layout the rank JSONL files use."""
    value = record.get(section, {}).get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main():
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    try:
        from examples.worldmodel.cosmos3_harness.training_metrics import (
            ASCEND_910B2_CUBE_PEAK_TFLOPS,
            compute_cosmos3_flops,
        )
    except ImportError:
        # The benchmark must run against whatever checkout the node has, which is not always the harness revision.
        # These are the same constants (cosmos3_harness/training_metrics.py): 20.34 TFLOP/sample forward,
        # x4 with full activation checkpointing, 33 TFLOP/sample VAE forward, 353.9 TFLOPS/NPU Cube peak.
        ASCEND_910B2_CUBE_PEAK_TFLOPS = 353.9

        def compute_cosmos3_flops(samples=1, activation_checkpointing=True, include_vae=False, **_):
            flops = samples * 20.34e12 * (4.0 if activation_checkpointing else 3.0)
            return flops + (samples * 33.0e12 if include_vae else 0.0)

    files = sorted(args.metrics_dir.glob("rank-*.jsonl"))
    if not files:
        raise SystemExit(f"no rank-*.jsonl under {args.metrics_dir}")
    ranks = [load_rank(path) for path in files]
    steps = sorted(set.intersection(*(set(rank) for rank in ranks)))
    if not steps:
        raise SystemExit("the rank files share no completed step")

    rows = []
    for step in steps:
        records = [rank[step] for rank in ranks]
        step_ms = max(number(record, "perf", "step_ms") for record in records)
        samples = int(sum(number(record, "data", "samples") for record in records))
        vae_ms = max(number(record, "data", "vae_ms") for record in records)
        fwd_ms = max(
            number(record, "perf", "forward_ms") or number(record, "perf", "fwd_ms") for record in records
        )
        bwd_ms = max(
            number(record, "perf", "backward_ms") or number(record, "perf", "bwd_ms") for record in records
        )
        optim_ms = max(number(record, "perf", "optim_ms") for record in records)
        wait_ms = max(number(record, "data", "wait_ms") for record in records)
        peak_gib = max(number(record, "mem", "peak_gib") for record in records)
        device_gib = max(number(record, "mem", "device_used_gib") for record in records)
        # Step 0 is the metrics module's warmup slot and partial writes have no perf record yet.
        if step_ms <= 0.0 or samples <= 0:
            continue
        rows.append(
            {
                "step": step,
                "step_s": step_ms / 1e3,
                "samples": samples,
                "samples_per_s": samples / (step_ms / 1e3) if step_ms else 0.0,
                "vae_s": vae_ms / 1e3,
                "fwd_s": fwd_ms / 1e3,
                "bwd_s": bwd_ms / 1e3,
                "optim_s": optim_ms / 1e3,
                "wait_s": wait_ms / 1e3,
                "peak_gib": peak_gib,
                "device_gib": device_gib,
                "mfu": compute_cosmos3_flops(
                    samples=samples,
                    activation_checkpointing=True,
                    include_vae=args.include_vae_flops,
                )
                / (step_ms / 1e3)
                / (args.world_size * ASCEND_910B2_CUBE_PEAK_TFLOPS * 1e12),
            }
        )

    steady = rows[args.drop_first :] or rows
    median = lambda key: statistics.median(row[key] for row in steady)
    summary = {
        "steps": len(rows),
        "steady_steps": len(steady),
        "step_s_median": median("step_s"),
        "step_s_min": min(row["step_s"] for row in steady),
        "samples_per_s": median("samples_per_s"),
        "samples_per_step": median("samples"),
        "vae_s_median": median("vae_s"),
        "fwd_s_median": median("fwd_s"),
        "bwd_s_median": median("bwd_s"),
        "optim_s_median": median("optim_s"),
        "wait_s_median": median("wait_s"),
        "peak_gib_max": max(row["peak_gib"] for row in steady),
        "device_gib_max": max(row["device_gib"] for row in steady),
        "mfu_median": median("mfu"),
    }
    summary["vae_share"] = summary["vae_s_median"] / summary["step_s_median"] if summary["step_s_median"] else 0.0

    if args.json:
        print(json.dumps(summary, indent=2))
        return
    print(f"steps {summary['steps']} (steady {summary['steady_steps']})")
    print(f"step       {summary['step_s_median']:.2f} s median ({summary['step_s_min']:.2f} min)")
    print(f"throughput {summary['samples_per_s']:.2f} samples/s | "
          f"{summary['samples_per_step']:.0f} samples/step")
    print(f"phases     vae {summary['vae_s_median']:.2f} s ({100 * summary['vae_share']:.1f}%) | "
          f"fwd {summary['fwd_s_median']:.2f} | bwd {summary['bwd_s_median']:.2f} | "
          f"optim {summary['optim_s_median']:.2f} | wait {summary['wait_s_median']:.2f}")
    print(f"memory     peak allocated {summary['peak_gib_max']:.2f} GiB | device used {summary['device_gib_max']:.2f} GiB")
    print(f"MFU        {100 * summary['mfu_median']:.1f}%"
          + (" (includes VAE FLOPs)" if args.include_vae_flops else " (transformer only)"))
    steady_steps = [row["step"] for row in steady]
    print("per-rank steady state:")
    for path, rank_records in zip(files, ranks):
        rank_rows = [rank_records[step] for step in steady_steps if step in rank_records]
        if not rank_rows:
            continue
        rank_step = statistics.median(number(row, "perf", "step_ms") for row in rank_rows) / 1e3
        rank_peak = max(number(row, "mem", "peak_gib") for row in rank_rows)
        rank_device = max(number(row, "mem", "device_used_gib") for row in rank_rows)
        rank_vae = statistics.median(number(row, "data", "vae_ms") for row in rank_rows) / 1e3
        print(f"  {path.stem}: step {rank_step:6.2f} s | vae {rank_vae:6.2f} s | "
              f"peak {rank_peak:6.2f} GiB | device {rank_device:6.2f} GiB")


if __name__ == "__main__":
    main()

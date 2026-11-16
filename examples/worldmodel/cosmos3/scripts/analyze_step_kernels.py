"""Attribute one profiled training step: compute vs communication, and how much of the wall is idle.

``train_sft.py --profile-step N`` dumps a torch_npu profile; this reads its ``kernel_details.csv`` and answers the
question the phase timers cannot: of the backward's ~11 s, how much is Cube work, how much is HCCL, and how
much is the device waiting between kernels.

    python scripts/analyze_step_kernels.py <profile-dir-with-ASCEND_PROFILER_OUTPUT> [--top 25]
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

CATEGORIES = (
    ("fused_kernel", re.compile(r"^main_kernel$", re.I)),
    (
        "communication",
        re.compile(
            r"hcom|hccl|allgather|reducescatter|reduce_scatter|allreduce|all_reduce|alltoall|broadcast|send|recv", re.I
        ),
    ),
    ("matmul", re.compile(r"matmul|gemm|batchmatmul|\bmm\b|addmm", re.I)),
    ("conv", re.compile(r"conv", re.I)),
    ("norm", re.compile(r"norm|rsqrt|rstd", re.I)),
    ("activation", re.compile(r"silu|swish|sigmoid|gelu|relu|tanh|exp\b", re.I)),
    ("softmax", re.compile(r"softmax", re.I)),
    ("reduce", re.compile(r"reduce|sum|mean|max\b|min\b", re.I)),
    (
        "elementwise",
        re.compile(
            r"mul|add|sub|div|pow|clamp|where|abs|sqrt|neg|cast|transdata|transpose|permute|slice|pad|concat|cat\b|copy|memcpy|move|fill|index|gather|scatter|select",
            re.I,
        ),
    ),
)


def category(name: str) -> str:
    for label, pattern in CATEGORIES:
        if pattern.search(name):
            return label
    return "other"


def column(header, *names):
    for name in names:
        for index, value in enumerate(header):
            if value.strip().lower() == name:
                return index
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    csv_path = None
    for candidate in (
        args.profile_dir / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv",
        *args.profile_dir.glob("*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"),
    ):
        if candidate.exists():
            csv_path = candidate
            break
    if csv_path is None:
        raise SystemExit(f"no kernel_details.csv under {args.profile_dir}")

    rows = list(csv.DictReader(open(csv_path)))
    header = list(rows[0].keys())
    name_i = column(header, "name", "op type", "op_type")
    dur_i = column(header, "duration(us)", "task duration(us)", "total time(us)")
    start_i = column(header, "start time(us)", "start(us)")
    step_i = column(header, "step id", "step")

    if step_i is not None:
        steps = defaultdict(float)
        for row in rows:
            steps[row[header[step_i]]] += float(row[header[dur_i]])
        print("[steps] " + ", ".join(f"step {k}: {v / 1e6:.2f} s" for k, v in sorted(steps.items())))

    per_category = defaultdict(float)
    per_category_calls = defaultdict(int)
    per_kernel = defaultdict(lambda: [0, 0.0])
    total_us = 0.0
    starts = []
    ends = []
    for row in rows:
        name = row[header[name_i]]
        duration = float(row[header[dur_i]])
        total_us += duration
        per_category[category(name)] += duration
        per_category_calls[category(name)] += 1
        entry = per_kernel[name]
        entry[0] += 1
        entry[1] += duration
        if start_i is not None:
            start = float(row[header[start_i]])
            starts.append(start)
            ends.append(start + duration)

    wall = (max(ends) - min(starts)) if starts else total_us
    print(
        f"[device] {len(rows)} kernels, busy {total_us / 1e6:.2f} s over wall {wall / 1e6:.2f} s "
        f"({100 * total_us / wall:.1f}% busy, {100 * (1 - total_us / wall):.1f}% idle)"
    )
    print(f"{'category':16s} {'calls':>7s} {'total_s':>9s} {'pct':>7s}")
    for label, value in sorted(per_category.items(), key=lambda kv: -kv[1]):
        print(f"{label:16s} {per_category_calls[label]:7d} {value / 1e6:9.3f} {100 * value / total_us:7.1f}")
    print(f"{'top kernels':60s} {'calls':>6s} {'total_s':>9s} {'pct':>7s}")
    for name, (calls, value) in sorted(per_kernel.items(), key=lambda kv: -kv[1][1])[: args.top]:
        print(f"{name[:60]:60s} {calls:6d} {value / 1e6:9.3f} {100 * value / total_us:7.1f}")


if __name__ == "__main__":
    main()

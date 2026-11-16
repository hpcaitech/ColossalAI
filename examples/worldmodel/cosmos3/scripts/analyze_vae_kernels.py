"""Roll up the Ascend profiler's per-kernel CSV into the numbers the VAE plan needs.

``torch_npu.profiler`` writes ``ASCEND_PROFILER_OUTPUT/kernel_details.csv`` next to the trace: one row per device
kernel with its duration and which engine ran it (AI_CORE is the Cube, AI_VECTOR the vector unit, AI_CPU the scalar
fallback). Plain ``torch.profiler`` records nothing on this stack, so this file, not ``key_averages``, is where the
encode's device time actually lives::

    python scripts/analyze_vae_kernels.py /home/niuzifan/logs/vae-deep/contiguous-float32-fp32

The engine split is the first thing to look at. Time on AI_CPU (or on AI_VECTOR for a convolution) is the part a
math-mode or layout change is going to move, and it bounds what operator fusion can give.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

CATEGORIES = (
    ("conv3d", re.compile(r"conv3d|conv_3d|Conv3D", re.I)),
    ("conv2d", re.compile(r"conv2d|conv_2d|Conv2D", re.I)),
    ("conv_other", re.compile(r"conv", re.I)),
    ("matmul", re.compile(r"matmul|gemm|mm\b", re.I)),
    ("norm", re.compile(r"norm|l2normal|rsqrt", re.I)),
    ("activation", re.compile(r"silu|swish|sigmoid|gelu|relu", re.I)),
    ("reduce", re.compile(r"reduce|sum|mean|max", re.I)),
    ("copy", re.compile(r"copy|data_copy|memcpy|transdata|transpose|permute", re.I)),
    ("pad_cat", re.compile(r"pad|concat|cat\b|strided_slice|slice|split|stack", re.I)),
    ("cast", re.compile(r"cast|to_bf16|to_fp|convert", re.I)),
    ("elementwise", re.compile(r"mul|add|sub|div|pow|clamp|where|abs|sqrt", re.I)),
)


def category(name: str) -> str:
    for label, pattern in CATEGORIES:
        if pattern.search(name):
            return label
    return "other"


def columns(header):
    """Map the columns this script needs regardless of the CANN export's exact naming."""
    def find(*names):
        for name in names:
            for index, column in enumerate(header):
                if column.strip().lower() == name:
                    return index
        return None

    return {
        "name": find("name", "op type", "op_type"),
        "type": find("type", "accelerator core", "core type"),
        "duration": find("duration(us)", "task duration(us)", "total time(us)"),
        "start": find("start time(us)", "start(us)"),
        "step": find("step id", "step"),
    }


def read_rows(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        where = columns(header)
        if where["name"] is None or where["duration"] is None:
            raise SystemExit(f"{path}: no name/duration column in {header}")
        for row in reader:
            try:
                duration = float(row[where["duration"]])
                start = float(row[where["start"]]) if where["start"] is not None else 0.0
            except (ValueError, IndexError):
                continue
            step = row[where["step"]].strip() if where["step"] is not None else ""
            yield row[where["name"]], (row[where["type"]] if where["type"] is not None else "?"), duration, start, step


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profile_dir", help="Directory holding npu-*/ASCEND_PROFILER_OUTPUT.")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--csv", help="Write the per-kernel rollup here.")
    args = parser.parse_args()

    root = Path(args.profile_dir)
    outputs = sorted(root.rglob("ASCEND_PROFILER_OUTPUT"))
    if not outputs:
        raise SystemExit(f"{root}: no ASCEND_PROFILER_OUTPUT directory")
    output = outputs[-1]
    kernel_csv = output / "kernel_details.csv"
    print(f"[profile] {output}")
    for name in sorted(p.name for p in output.iterdir()):
        size = (output / name).stat().st_size / 2**10
        print(f"[profile]   {name:<34}{size:>10.0f} KiB")

    if not kernel_csv.exists():
        raise SystemExit(f"{kernel_csv} missing; only the trace was written")

    per_name = defaultdict(lambda: [0, 0.0])
    per_engine = defaultdict(lambda: [0, 0.0])
    total = 0.0
    rows = 0
    raw = list(read_rows(kernel_csv))
    # Rows the profiler could not attribute to a step are the tail of the warm-up call; the attributed step is the
    # one encode whose numbers are comparable with a wall-clock measurement.
    steps = {row[4] for row in raw if row[4]}
    if len(steps) == 1:
        step = steps.pop()
        raw = [row for row in raw if row[4] == step]
        print(f"\n[device] keeping step {step} only, {len(raw)} kernels")
    starts = []
    for name, engine, duration, start, _ in raw:
        rows += 1
        total += duration
        per_name[name][0] += 1
        per_name[name][1] += duration
        per_engine[engine][0] += 1
        per_engine[engine][1] += duration
        starts.append((start, duration))
    print(f"\n[device] {rows} kernels, {total / 1e6:.3f}s summed")

    # The export can hold more than one call of the profiled step; a long gap between kernels is where one ends and
    # the next begins, and the per-call number is the only one comparable with a wall-clock measurement.
    starts.sort()
    bursts, begin, previous_end, count, busy = [], starts[0][0], starts[0][0], 0, 0.0
    for start, duration in starts:
        if start - previous_end > 50_000:
            bursts.append((begin, previous_end, count, busy))
            begin, count, busy = start, 0, 0.0
        count += 1
        busy += duration
        previous_end = max(previous_end, start + duration)
    bursts.append((begin, previous_end, count, busy))
    print(f"[device] {len(bursts)} burst(s) at >50ms of idle:")
    for begin, end, count, busy in bursts:
        print(f"[device]   {count:>5} kernels  wall {(end - begin) / 1e6:>6.3f}s  busy {busy / 1e6:>6.3f}s"
              f"  utilisation {100 * busy / (end - begin):>5.1f}%")

    print(f"\n{'engine':<16}{'kernels':>9}{'total_s':>10}{'pct':>7}")
    for engine, (count, seconds) in sorted(per_engine.items(), key=lambda kv: -kv[1][1]):
        print(f"{engine:<16}{count:>9}{seconds / 1e6:>10.4f}{100 * seconds / total if total else 0:>6.1f}%")

    ranked = sorted(per_name.items(), key=lambda kv: -kv[1][1])
    print(f"\n[device] top {args.top} kernels")
    print(f"{'kernel':<56}{'calls':>7}{'total_s':>10}{'pct':>7}{'mean_us':>10}")
    for name, (count, seconds) in ranked[: args.top]:
        print(f"{name[:55]:<56}{count:>7}{seconds / 1e6:>10.4f}"
              f"{100 * seconds / total if total else 0:>6.1f}%{seconds / count:>10.1f}")

    per_category = defaultdict(lambda: [0, 0.0])
    for name, (count, seconds) in per_name.items():
        row = per_category[category(name)]
        row[0] += count
        row[1] += seconds
    print("\n[device] by category")
    print(f"{'category':<14}{'calls':>8}{'total_s':>10}{'pct':>7}")
    for label, (count, seconds) in sorted(per_category.items(), key=lambda kv: -kv[1][1]):
        print(f"{label:<14}{count:>8}{seconds / 1e6:>10.4f}{100 * seconds / total if total else 0:>6.1f}%")

    if args.csv:
        with open(args.csv, "w") as handle:
            handle.write("kernel,calls,total_s,category\n")
            for name, (count, seconds) in ranked:
                handle.write(f'"{name}",{count},{seconds / 1e6:.6f},{category(name)}\n')
        print(f"\n[csv] {args.csv}")


if __name__ == "__main__":
    main()

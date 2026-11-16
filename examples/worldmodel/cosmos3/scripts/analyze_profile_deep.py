"""Second-angle read of an Ascend profiler dump: idle, launch-bound kernels, chains, copies, host ops.

``analyze_vae_kernels.py`` answers "which kernel is expensive". This answers the questions the next optimization
needs: where the device is *idle* between kernels, how much time is spent in kernels too small to pay for their own
launch, which memory-bound kernels sit back to back on the same tensor (fusion candidates), and how much of the
encode is layout traffic (copy/transpose/cat/pad) rather than math.

    python scripts/analyze_profile_deep.py /tmp/prof_norm/npu-*/ASCEND_PROFILER_OUTPUT

Only the profiler's own CSVs are read; no torch, no device.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

CATEGORIES = (
    ("conv", re.compile(r"conv", re.I)),
    ("matmul", re.compile(r"matmul|gemm|\bmm\b|batchmatmul", re.I)),
    ("norm", re.compile(r"norm|l2norm|rsqrt|rms", re.I)),
    ("activation", re.compile(r"silu|swish|sigmoid|gelu|relu|tanh", re.I)),
    ("reduce", re.compile(r"reduce|sum|mean|amax|max\b", re.I)),
    (
        "layout",
        re.compile(r"transpose|permute|TransposeAiCore|transdata|copy|cat\b|concat|pad|slice|split|stack", re.I),
    ),
    ("cast", re.compile(r"cast|to_bf16|to_fp|convert", re.I)),
    ("elementwise", re.compile(r"mul|add|sub|div|pow|clamp|where|abs|sqrt|exp|neg", re.I)),
    ("comm", re.compile(r"hcom|hccl|allreduce|reducescatter|allgather|broadcast", re.I)),
)


def category(name: str) -> str:
    for label, pattern in CATEGORIES:
        if pattern.search(name):
            return label
    return "other"


def read_kernels(path: Path):
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        fields = {key.strip().lower(): key for key in reader.fieldnames}
        name_key = fields.get("name")
        start_key = fields.get("start time(us)") or fields.get("start(us)")
        dur_key = fields.get("duration(us)") or fields.get("task duration(us)")
        step_key = fields.get("step id")
        rows = []
        for row in reader:
            try:
                rows.append(
                    (
                        row[name_key],
                        float(row[start_key]),
                        float(row[dur_key]),
                        (row[step_key].strip() if step_key else ""),
                    )
                )
            except (TypeError, ValueError):
                continue
    return rows


def read_operators(path: Path):
    """Host-side ops with their device self time; the shape column is empty on this CANN export."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        fields = {key.strip().lower(): key for key in reader.fieldnames}
        name_key = fields.get("name")
        dev_key = fields.get("device self duration(us)")
        host_key = fields.get("host self duration(us)")
        call_key = fields.get("call stack")
        rows = []
        for row in reader:
            try:
                dev = float(row[dev_key]) if dev_key else 0.0
            except (TypeError, ValueError):
                dev = 0.0
            try:
                host = float(row[host_key]) if host_key else 0.0
            except (TypeError, ValueError):
                host = 0.0
            rows.append((row.get(name_key, ""), dev, host, row.get(call_key, "") if call_key else ""))
    return rows


def union_busy(intervals):
    total = 0.0
    begin, end = None, None
    for start, stop in sorted(intervals):
        if begin is None:
            begin, end = start, stop
        elif start > end:
            total += end - begin
            begin, end = start, stop
        else:
            end = max(end, stop)
    if begin is not None:
        total += end - begin
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profile_dir", help="ASCEND_PROFILER_OUTPUT directory (or a parent of one).")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--small-us", type=float, default=10.0, help="A kernel at or below this is launch-bound.")
    parser.add_argument("--gap-us", type=float, default=20.0)
    parser.add_argument("--chains", type=int, default=12)
    args = parser.parse_args()

    root = Path(args.profile_dir)
    if not (root / "kernel_details.csv").exists():
        found = sorted(root.rglob("kernel_details.csv"))
        if not found:
            raise SystemExit(f"{root}: no kernel_details.csv")
        root = found[-1].parent
    rows = read_kernels(root / "kernel_details.csv")
    steps = {row[3] for row in rows if row[3]}
    if len(steps) == 1:
        keep = steps.pop()
        rows = [row for row in rows if row[3] == keep]
        print(f"[scope] step {keep}: {len(rows)} kernels")
    else:
        print(f"[scope] all rows: {len(rows)} kernels, steps={sorted(steps)[:6]}")

    per_name = defaultdict(lambda: [0, 0.0])
    total = 0.0
    for name, _start, duration, _step in rows:
        per_name[name][0] += 1
        per_name[name][1] += duration
        total += duration
    wall = rows and (max(s + d for _n, s, d, _ in rows) - min(s for _n, s, _d, _ in rows))
    busy = union_busy((s, s + d) for _n, s, d, _ in rows)
    span = wall or 1.0
    print(
        f"\n[idle] device busy {busy / 1e6:.3f}s of {span / 1e6:.3f}s span -> {100 * busy / span:.1f}% busy, "
        f"idle {(span - busy) / 1e6:.3f}s"
    )
    print(f"[idle] summed kernel time {total / 1e6:.3f}s (overlap inside {total / busy:.2f}x)")

    ordered = sorted(rows, key=lambda row: row[1])
    gaps = []
    previous_end = ordered[0][1] + ordered[0][2]
    previous_name = ordered[0][0]
    for name, start, duration, _step in ordered[1:]:
        if start > previous_end + args.gap_us:
            gaps.append((start - previous_end, previous_name, name, start))
        previous_end = max(previous_end, start + duration)
        previous_name = name
    if gaps:
        idle = sum(gap for gap, _a, _b, _s in gaps)
        print(f"\n[gaps] {len(gaps)} idle gaps > {args.gap_us:.0f}us, {idle / 1e6:.3f}s total")
        for gap, before, after, _start in sorted(gaps, reverse=True)[:12]:
            print(f"  {gap / 1e3:8.2f} ms  after {before[:44]:<46} before {after[:40]}")
    else:
        print(f"\n[gaps] none over {args.gap_us:.0f}us")

    small = sum(seconds for _n, (calls, seconds) in per_name.items() for _ in [0] if seconds / calls <= args.small_us)
    small_calls = sum(calls for _n, (calls, seconds) in per_name.items() if seconds / calls <= args.small_us)
    print(
        f"\n[launch] {small_calls} kernels <= {args.small_us:.0f}us mean: {small / 1e6:.3f}s "
        f"({100 * small / total:.1f}% of kernel time, {100 * small_calls / len(rows):.0f}% of calls)"
    )
    print("[launch] smallest-kernel names by call count")
    tiny = sorted(
        ((name, calls, seconds) for name, (calls, seconds) in per_name.items() if seconds / calls <= args.small_us),
        key=lambda kv: -kv[1],
    )
    for name, calls, seconds in tiny[:10]:
        print(f"  {calls:>5} calls  mean {seconds / calls:6.2f}us  total {seconds / 1e3:8.1f}ms  {name[:52]}")

    ranked = sorted(per_name.items(), key=lambda kv: -kv[1][1])
    print(f"\n[kernels] top {args.top} by total time")
    print(f"{'kernel':<58}{'calls':>7}{'total_ms':>10}{'pct':>7}{'mean_us':>10}")
    for name, (calls, seconds) in ranked[: args.top]:
        print(f"{name[:57]:<58}{calls:>7}{seconds / 1e3:>10.2f}{100 * seconds / total:>6.1f}%{seconds / calls:>10.1f}")

    per_category = defaultdict(lambda: [0, 0.0])
    for name, (calls, seconds) in per_name.items():
        row = per_category[category(name)]
        row[0] += calls
        row[1] += seconds
    print("\n[categories]")
    print(f"{'category':<14}{'calls':>8}{'total_ms':>11}{'pct':>7}{'mean_us':>10}")
    for label, (calls, seconds) in sorted(per_category.items(), key=lambda kv: -kv[1][1]):
        print(f"{label:<14}{calls:>8}{seconds / 1e3:>11.2f}{100 * seconds / total:>6.1f}%{seconds / calls:>10.1f}")

    # Chains: consecutive runs of the same category (no gaps) -- two memory-bound kernels back to back on the same
    # tensor are one fusion candidate; five in a row is what a decomposed layer-norm looks like.
    chains, current = [], []
    previous_end = None
    for name, start, duration, _step in ordered:
        if previous_end is not None and start - previous_end > args.gap_us:
            if len(current) > 1:
                chains.append(current)
            current = []
        current.append((name, duration))
        previous_end = max(previous_end or 0.0, start + duration)
    if len(current) > 1:
        chains.append(current)
    chains.sort(key=lambda chain: -sum(d for _n, d in chain))
    print(f"\n[chains] longest back-to-back runs (fusion candidates), showing {args.chains}")
    for chain in chains[: args.chains]:
        names = Counter(name for name, _d in chain)
        label = ", ".join(f"{name[:26]}x{count}" for name, count in names.most_common(4))
        print(f"  {sum(d for _n, d in chain) / 1e3:9.2f} ms  {len(chain):>4} kernels  {label}")

    # Elementwise runs: consecutive memory-bound kernels with no gap between them are one fusion candidate. A run of
    # two Add/Mul kernels over the same activation costs three full passes where one fused pass would do.
    runs = []
    current = []
    previous_end = None
    for name, start, duration, _step in ordered:
        label = category(name)
        if previous_end is not None and start - previous_end > args.gap_us:
            if current:
                runs.append(current)
            current = []
        if current and category(current[0][0]) != label:
            runs.append(current)
            current = []
        current.append((name, duration))
        previous_end = max(previous_end or 0.0, start + duration)
    if current:
        runs.append(current)
    elementwise = [run for run in runs if category(run[0][0]) == "elementwise"]
    histogram = Counter(min(len(run), 8) for run in elementwise)
    print("\n[elementwise] runs of back-to-back elementwise kernels (fusion candidates)")
    print(f"{'run length':<12}{'runs':>8}{'kernels':>9}{'total_ms':>11}")
    for length in sorted(histogram):
        subset = [run for run in elementwise if min(len(run), 8) == length]
        print(
            f"{('>=8' if length == 8 else str(length)):<12}{len(subset):>8}"
            f"{sum(len(run) for run in subset):>9}{sum(d for run in subset for _n, d in run) / 1e3:>11.2f}"
        )
    fused_ms = sum(d for run in elementwise for _n, d in run if len(run) >= 2)
    print(
        f"[elementwise] {fused_ms / 1e3:.2f} ms in runs of >=2 kernels; a pairwise fusion removes one pass per "
        f"boundary, so the ceiling is about {fused_ms / 1e3 / 2:.2f} ms"
    )
    patterns = Counter(tuple(name for name, _d in run) for run in elementwise if len(run) >= 2)
    for pattern, count in patterns.most_common(6):
        seconds = sum(d for run in elementwise for _n, d in run if tuple(name for name, _d in run) == pattern)
        print(f"  {count:>4}x {seconds / 1e3:8.2f} ms  {' -> '.join(name[:22] for name in pattern[:5])}")

    operators = read_operators(root / "operator_details.csv")
    if operators:
        per_op = defaultdict(lambda: [0, 0.0, 0.0])
        for name, dev, host, _stack in operators:
            row = per_op[name]
            row[0] += 1
            row[1] += dev
            row[2] += host
        host_total = sum(row[2] for row in per_op.values())
        print("\n[host] top host ops by device self time")
        print(f"{'op':<44}{'calls':>7}{'dev_ms':>10}{'host_ms':>10}{'dev/call_us':>12}")
        for name, (calls, dev, host) in sorted(per_op.items(), key=lambda kv: -kv[1][1])[:20]:
            print(f"{name[:43]:<44}{calls:>7}{dev / 1e3:>10.2f}{host / 1e3:>10.2f}{dev / max(calls, 1):>12.1f}")
        print(f"[host] summed host self time {host_total / 1e3:.1f} ms over {len(operators)} op records")
        allocs = per_op.get("empty_tensor", [0, 0.0, 0.0])
        print(f"[host] empty_tensor {allocs[0]} calls, host {allocs[2] / 1e3:.1f} ms")
        layout_host = sum(row[2] for name, row in per_op.items() if category(name) == "layout")
        print(
            f"[host] layout-class host time {layout_host / 1e3:.1f} ms "
            f"({100 * layout_host / host_total if host_total else 0:.1f}% of host)"
        )


if __name__ == "__main__":
    main()

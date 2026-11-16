"""Summarise a ``torch.profiler`` chrome trace of the VAE encode without loading it into a browser.

The trace is the only record that has every kernel, not just the top thirty rows ``key_averages`` prints, so this
script answers the questions the optimisation plan needs: what fraction of wall time the device is actually busy,
which kernels carry the time, and how much of it is elementwise work that a fused or a wider layout should erase.

It reads the JSON straight off disk (traces of this size do not fit in a browser tab comfortably either) and streams
it, so a several-hundred-megabyte file is fine::

    python scripts/analyze_vae_trace.py /home/niuzifan/logs/vae-deep/trace.json --top 30

Categories come from the kernel names, which is crude but stable across CANN versions: a name containing ``Conv3D``
is a 3D convolution whatever the vendor calls the rest of it.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

# Order matters: the first pattern that matches names the category.
CATEGORIES = (
    ("conv3d", re.compile(r"conv3d|Conv3D|Conv3d", re.I)),
    ("conv2d", re.compile(r"conv2d|Conv2D|Conv2d", re.I)),
    ("conv_other", re.compile(r"conv", re.I)),
    ("matmul", re.compile(r"matmul|mm\b|gemm|batch_norm_matmul", re.I)),
    ("norm", re.compile(r"norm|l2|rsqrt|sqrt|RmsNorm", re.I)),
    ("activation", re.compile(r"silu|swish|sigmoid|gelu|relu|activation", re.I)),
    ("reduce", re.compile(r"reduce|sum|mean|max|argmax", re.I)),
    ("softmax", re.compile(r"softmax", re.I)),
    ("copy", re.compile(r"copy|memcpy|DataCopy|d2d|h2d|d2h", re.I)),
    ("pad_cat", re.compile(r"pad|cat|concat|split|slice|stack", re.I)),
    ("cast", re.compile(r"cast|to\b|convert", re.I)),
    ("transpose", re.compile(r"transpose|permute|contiguous|as_strided|view", re.I)),
    ("elementwise", re.compile(r"mul|add|sub|div|pow|clamp|where", re.I)),
)


def category(name: str) -> str:
    for label, pattern in CATEGORIES:
        if pattern.search(name):
            return label
    return "other"


def load_events(path):
    with open(path) as handle:
        data = json.load(handle)
    events = data["traceEvents"] if isinstance(data, dict) else data
    return [e for e in events if e.get("ph") == "X" and e.get("dur")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--csv", help="Also write the per-kernel table here.")
    args = parser.parse_args()

    events = load_events(args.trace)
    # The device side is the pid whose events carry the kernel category; torch_npu splits it per stream under one pid.
    stats = defaultdict(lambda: [0, 0])
    for event in events:
        row = stats[event.get("pid")]
        row[0] += 1
        row[1] += "kernel" in str(event.get("cat", "")).lower()
    print(f"[trace] {len(events)} complete events over {len(stats)} pid(s): "
          + ", ".join(f"{pid}: {n} ({k} kernel)" for pid, (n, k) in stats.items()))
    device_pid = max(stats, key=lambda pid: (stats[pid][1], stats[pid][0]))
    device = [e for e in events if e.get("pid") == device_pid]
    print(f"[trace] device pid {device_pid} with {len(device)} events")

    # Busy time and span: overlap between streams is real on this device, so busy can exceed wall.
    span_lo = min(e["ts"] for e in device)
    span_hi = max(e["ts"] + e["dur"] for e in device)
    wall = (span_hi - span_lo) / 1e6
    busy = sum(e["dur"] for e in device) / 1e6
    print(f"[device] wall span {wall:.3f}s, summed kernel time {busy:.3f}s ({100 * busy / wall:.1f}% of span)")
    per_stream = defaultdict(float)
    for e in device:
        per_stream[e.get("tid")] += e["dur"]
    print("[device] per-stream " + ", ".join(f"{tid}: {d / 1e6:.3f}s" for tid, d in
                                             sorted(per_stream.items(), key=lambda kv: -kv[1])[:6]))

    per_kernel = defaultdict(lambda: [0, 0.0, 0.0])
    for e in device:
        row = per_kernel[e["name"]]
        row[0] += 1
        row[1] += e["dur"] / 1e6
        row[2] = max(row[2], e["dur"] / 1e6)
    rows = sorted(per_kernel.items(), key=lambda kv: -kv[1][1])
    total = busy or 1.0

    print(f"\n[device] top {args.top} kernels by total time")
    print(f"{'kernel':<58}{'calls':>7}{'total_s':>10}{'pct':>7}{'mean_us':>10}{'max_s':>9}")
    for name, (calls, seconds, peak) in rows[: args.top]:
        print(f"{name[:57]:<58}{calls:>7}{seconds:>10.4f}{100 * seconds / total:>6.1f}%"
              f"{1e6 * seconds / calls:>10.1f}{peak:>9.4f}")

    print("\n[device] by category")
    cat_rows = defaultdict(lambda: [0, 0.0])
    for name, (calls, seconds, _) in per_kernel.items():
        row = cat_rows[category(name)]
        row[0] += calls
        row[1] += seconds
    print(f"{'category':<14}{'calls':>8}{'total_s':>10}{'pct':>7}")
    for label, (calls, seconds) in sorted(cat_rows.items(), key=lambda kv: -kv[1][1]):
        print(f"{label:<14}{calls:>8}{seconds:>10.4f}{100 * seconds / total:>6.1f}%")

    if args.csv:
        with open(args.csv, "w") as handle:
            handle.write("kernel,calls,total_s,max_s,category\n")
            for name, (calls, seconds, peak) in rows:
                handle.write(f'"{name}",{calls},{seconds:.6f},{peak:.6f},{category(name)}\n')
        print(f"\n[csv] {args.csv} ({len(rows)} kernels)")


if __name__ == "__main__":
    main()

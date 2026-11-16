"""Per-step spread of the 48-rank run: who arrives last at the end-of-backward collective, and what it costs.

Every rank records ``data.wait_ms`` (time it stood waiting for its batch) and ``perf.backward_ms`` (the backward
window, which contains the gradient collective).  In a replica group the world's backward only ends when the slowest
rank gets there, so the interesting numbers per step are:

* ``bwd_min`` - the fastest rank's backward: roughly pure compute, no waiting at the collective;
* ``bwd_max`` - the slowest rank's backward, i.e. what the step actually paid;
* ``spread`` = ``bwd_max - bwd_min``, the exposed wait the fast ranks spent inside their backward;
* ``max_wait`` and the rank carrying it, which is the usual explanation for a spread that jumps by 10 s.

Usage (metrics directories from all nodes in one place, see ``pull_6node_metrics.sh``)::

    python analyze_multinode_spread.py /tmp/6node_metrics --since 3

Each ``metrics`` directory holds 8 ``rank-*.jsonl`` files; pass any number of them, or a parent directory that
contains several ``metrics`` directories (the per-node copies of the same run).
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path


def load(root: Path) -> dict[int, dict[int, dict]]:
    """step -> rank -> record fields we care about."""
    files = sorted(root.rglob("rank-*.jsonl"))
    steps: dict[int, dict[int, dict]] = {}
    for path in files:
        for line in path.read_text(errors="ignore").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") != "record" or not record.get("step"):
                continue
            data = record.get("data") or {}
            perf = record.get("perf") or {}
            steps.setdefault(int(record["step"]), {})[int(record.get("rank", 0))] = {
                "wait": float(data.get("wait_ms") or 0.0) / 1e3,
                "vae": float(data.get("vae_ms") or 0.0) / 1e3,
                "fwd": float(perf.get("forward_ms") or 0.0) / 1e3,
                "bwd": float(perf.get("backward_ms") or 0.0) / 1e3,
                "step": float(perf.get("step_ms") or 0.0) / 1e3,
                "loss": float((record.get("loss") or {}).get("total") or float("nan")),
            }
    return steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="directory holding the run's per-node metrics copies")
    parser.add_argument("--since", type=int, default=3, help="first step to include (drops the cold start)")
    parser.add_argument("--ranks-per-node", type=int, default=8)
    args = parser.parse_args()

    steps = load(args.root)
    if not steps:
        raise SystemExit(f"no rank-*.jsonl records under {args.root}")
    print(f"{len(steps)} steps, {max(len(v) for v in steps.values())} ranks in the widest step")

    print("\n step  ranks  step_s   bwd_min bwd_med bwd_max  spread  max_wait  who")
    steady_spread, steady_step, wait_by_node = [], [], {}
    for step in sorted(steps):
        if step < args.since:
            continue
        rows = steps[step]
        if len(rows) < 2:
            continue
        bwd = [r["bwd"] for r in rows.values()]
        waits = {rank: r["wait"] for rank, r in rows.items()}
        laggard = max(waits, key=waits.get)
        spread = max(bwd) - min(bwd)
        steady_spread.append(spread)
        steady_step.append(max(r["step"] for r in rows.values()))
        for rank, r in rows.items():
            wait_by_node.setdefault(rank // args.ranks_per_node, []).append(r["wait"])
        print(
            f"{step:5d} {len(rows):6d} {stats.median([r['step'] for r in rows.values()]):7.2f}"
            f" {min(bwd):8.2f} {stats.median(bwd):7.2f} {max(bwd):7.2f} {spread:7.2f}"
            f" {waits[laggard]:9.2f}  rank {laggard}"
        )

    print("\nper node (rank // 8):")
    for node in sorted(wait_by_node):
        waits = wait_by_node[node]
        print(
            f"  node_rank {node}: median data wait {stats.median(waits):5.2f}s   p90 {sorted(waits)[int(len(waits) * 0.9)]:5.2f}s"
        )

    if steady_spread:
        print(
            f"\nsteady median spread {stats.median(steady_spread):.2f}s "
            f"= {100 * stats.median(steady_spread) / stats.median(steady_step):.1f}% of the "
            f"{stats.median(steady_step):.2f}s step (p90 spread {sorted(steady_spread)[int(len(steady_spread) * 0.9)]:.2f}s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

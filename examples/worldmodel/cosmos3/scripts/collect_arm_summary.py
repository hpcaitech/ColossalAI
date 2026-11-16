"""One table over every benchmark arm in the log directory: steady-state step, phases, memory.

Each ``bench_online_step*.sh`` run writes ``[step N] loss ... step X s (forward F, backward B, optim O) ...
data (vae V, wait W) ... mem M/61.0 GiB`` lines. Step 1 is the cold start and is excluded; the rest are summarised by
median, which is what the arm-to-arm comparisons in the write-up quote.

    python scripts/collect_arm_summary.py /mnt/disk0/jiangzs/nzf/bench_online/logs

Arms that never produced a steady step (compile failures, OOM) are listed with the number of steps they managed, so a
failed experiment cannot be mistaken for a missing one.
"""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

STEP = re.compile(
    r"\[step (\d+)\] loss [\d.]+ \(v [\d.]+ a [\d.]+\) \| (?:samples \d+ \| )?step ([\d.]+)s "
    r"\(forward ([\d.]+), backward ([\d.]+), optim ([\d.]+)\)"
    r".*?data \(vae ([\d.]+), wait ([\d.]+)\).*?mem ([\d.]+)/"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", type=Path)
    parser.add_argument("--min-steps", type=int, default=2, help="Steady steps required to list an arm.")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.logs.glob("*.log")):
        steady = [m.groups() for m in STEP.finditer(path.read_text(errors="replace")) if int(m.group(1)) > 1]
        if len(steady) < args.min_steps:
            if steady or path.stat().st_size:
                rows.append((path.stem, len(steady), None))
            continue
        columns = list(zip(*steady))
        values = [[float(value) for value in column] for column in columns[1:]]
        medians = [statistics.median(column) for column in values]
        rows.append((path.stem, len(steady), medians))

    print(f"{'arm':<22}{'steps':>6}{'step':>9}{'fwd':>8}{'bwd':>8}{'optim':>8}{'vae':>8}{'wait':>8}{'mem':>8}")
    for name, count, medians in sorted(rows, key=lambda row: (row[2] is None, row[2][0] if row[2] else 9e9)):
        if medians is None:
            print(f"{name:<22}{count:>6}{'  (no steady step)':>40}")
            continue
        step, fwd, bwd, optim, vae, wait, mem = medians
        print(f"{name:<22}{count:>6}{step:>9.2f}{fwd:>8.2f}{bwd:>8.2f}{optim:>8.2f}{vae:>8.2f}{wait:>8.2f}{mem:>8.1f}")


if __name__ == "__main__":
    main()

"""Correlate the step-level dataloader wait with page-cache refill and disk reads.

``io_sampler.sh`` records ``/proc/vmstat`` and ``/proc/diskstats`` once a second; ``rank-*.jsonl`` records when each
step ended and how long it waited for data. This joins the two: for every step, the IO counters over the window that
ends at that step tell whether the wait was a disk stall (major faults + read sectors spiking) or something else.

    python scripts/analyze_io_sample.py io_sample.csv rank-000.jsonl
"""

from __future__ import annotations

import json
import sys


def main(csv_path: str, metrics_path: str) -> None:
    rows = []
    with open(csv_path) as handle:
        next(handle)
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) != 7:
                continue
            rows.append([float(value) for value in parts])
    if len(rows) < 3:
        raise SystemExit("not enough samples")

    steps = []
    with open(metrics_path) as handle:
        for line in handle:
            record = json.loads(line)
            when = record.get("time")
            data = record.get("data") or {}
            if when is None or "wait_ms" not in data:
                continue
            steps.append((float(when), record.get("step"), float(data["wait_ms"]), float(data.get("vae_ms", 0.0))))

    # Per-sample deltas: [faults, pages in, dm0 sectors, nvme sectors]
    deltas = []
    for previous, current in zip(rows, rows[1:]):
        deltas.append(
            (
                current[0],
                current[1] - previous[1],
                current[2] - previous[2],
                (current[4] - previous[4]) + (current[5] - previous[5]) + (current[6] - previous[6]),
                current[3],
            )
        )

    spikes = sorted(deltas, key=lambda item: -item[3])[:5]
    print("[io] top 5 seconds by disk read sectors (whole run)")
    for when, faults, pgin, sectors, avail in spikes:
        print(
            f"  t={when:.0f} read {sectors:>9.0f} sectors  major_faults {faults:>7.0f}  pgin {pgin:>9.0f}  "
            f"MemAvailable {avail / 2**20:.1f} GiB"
        )

    print("\n[io] per step, the window that ends at the step record (15 s before it)")
    print(f"{'step':>5}{'wait_ms':>10}{'vae_ms':>10}{'major_faults/s':>16}{'read MiB/s':>12}{'MemAvail GiB':>14}")
    for when, step, wait_ms, vae_ms in steps:
        window = [item for item in deltas if when - 15.0 <= item[0] <= when]
        if not window:
            continue
        faults = sum(item[1] for item in window) / len(window)
        sectors = sum(item[3] for item in window) / len(window)
        avail = min(item[4] for item in window) / 2**20
        print(f"{step:>5}{wait_ms:>10.0f}{vae_ms:>10.0f}{faults:>16.0f}{sectors * 512 / 2**20:>12.1f}{avail:>14.1f}")

    overall = sum(item[3] for item in deltas) / max(len(deltas), 1)
    print(f"\n[io] run average read {overall * 512 / 2**20:.1f} MiB/s over {len(deltas)} samples")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

"""Merge the per-rank host-probe dumps and print the hottest Python call sites."""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict


def main(pattern: str) -> None:
    per_kind = defaultdict(lambda: [0, 0.0])
    files = sorted(glob.glob(pattern))
    ranks = 0
    for path in files:
        try:
            rows = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        if not rows:
            continue
        ranks += 1
        for row in rows:
            key = (row["kind"], row["site"])
            per_kind[key][0] += row["calls"]
            per_kind[key][1] += row["us"]

    print(f"[hostprobe] {ranks} rank file(s) of {len(files)}")
    print(f"{'kind':<8}{'calls':>8}{'ms':>10}  site")
    for (kind, site), (calls, micros) in sorted(per_kind.items(), key=lambda kv: -kv[1][1])[:20]:
        print(f"{kind:<8}{calls:>8}{micros / 1000:>10.1f}  {site}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "hostprobe.json.*.json")

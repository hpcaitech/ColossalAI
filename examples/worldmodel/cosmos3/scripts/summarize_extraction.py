"""Steady-state throughput from the extraction logs' batch counters.

Each rank prints ``batches=<n>/<total> samples=<s> at=<t>s`` every few batches, where ``t`` is measured from
just after the VAE load. The first batch of a run carries the one-off kernel warmup, so the slope between the
first and last counter is the number that extrapolates to a multi-hour run.
"""

import re
import statistics
import sys
from collections import defaultdict

COUNTER = re.compile(r"batches=(\d+)/(\d+) samples=(\d+) at=([0-9.]+)s")

TOTAL_WINDOWS = 20_109_000  # unique windows in the full DROID split, see draft doc section 8
KEEP_RANGES_SHARE = 0.74

slopes = defaultdict(list)
finished = defaultdict(int)
for line in open(sys.argv[1]):
    if "|" not in line:
        continue
    tag, rank, rest = line.split("|", 2)
    points = [(int(s), float(t)) for _, _, s, t in COUNTER.findall(rest)]
    if len(points) >= 2:
        first_sample, first_time = points[0]
        last_sample, last_time = points[-1]
        if last_sample > first_sample:
            slopes[tag].append((last_time - first_time) / (last_sample - first_sample))
    if points:
        finished[tag] += 1

print(f"{'case':10s} {'ranks':>5s} {'s/sample':>9s} {'node/s':>7s} {'6 nodes/s':>9s} {'20.11M days':>12s} {'keep74%':>8s}")
for tag in sorted(slopes):
    per_sample = statistics.mean(slopes[tag])
    node_rate = 8 / per_sample
    six_nodes = 6 * node_rate
    full_days = TOTAL_WINDOWS / six_nodes / 86400
    print(
        f"{tag:10s} {len(slopes[tag]):5d} {per_sample:9.3f} {node_rate:7.2f} {six_nodes:9.1f} "
        f"{full_days:12.2f} {full_days * KEEP_RANGES_SHARE:8.2f}"
    )

vae_floor = 0.47
print()
print(f"VAE-only floor at 0.47 s/sample: {TOTAL_WINDOWS * vae_floor / 48 / 86400:.2f} days on 48 cards")

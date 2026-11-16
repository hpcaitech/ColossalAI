"""Aggregate a multi-node extraction run from the per-rank batch counters.

Both dumps have one line per rank: ``node|rank-XX|batches=1/32 samples=4 at=66.5s|batches=5/32 ...``.
The steady-state slope between the first and the last counter excludes the one-off kernel warmup, which is
60-70 s here and would otherwise dominate a 2.5-minute run.
"""

import re
import statistics
import sys

COUNTER = re.compile(r"batches=(\d+)/(\d+) samples=(\d+) at=([0-9.]+)s")

per_node = {}
for path in sys.argv[1:]:
    for line in open(path):
        parts = line.strip().split("|")
        if len(parts) < 3 or not parts[0].startswith("node"):
            continue
        node, rank, rest = parts[0], parts[1], "|".join(parts[2:])
        points = [(int(s), float(t)) for _, _, s, t in COUNTER.findall(rest)]
        if len(points) < 2:
            continue
        (s0, t0), (s1, t1) = points[0], points[-1]
        if s1 <= s0:
            continue
        per_node.setdefault(node, []).append((t1 - t0) / (s1 - s0))

print(f"{'node':8s} {'ranks':>5s} {'s/sample':>9s} {'node samples/s':>15s} {'MB/s':>7s}")
total_ranks = 0
all_slopes = []
for node in sorted(per_node):
    slopes = per_node[node]
    mean = statistics.mean(slopes)
    total_ranks += len(slopes)
    all_slopes.extend(slopes)
    print(f"{node:8s} {len(slopes):5d} {mean:9.3f} {len(slopes) / mean:15.2f} {len(slopes) / mean * 1.09:7.1f}")

mean = statistics.mean(all_slopes)
print()
print(
    f"all {total_ranks} ranks: mean {mean:.3f} s/sample -> {total_ranks / mean:.2f} samples/s "
    f"({total_ranks / mean * 1.09:.1f} MB/s of latents)"
)

TRAIN_SAMPLES_PER_STEP = 1024  # 4 nodes x 8 cards x 32 samples, the split the question asks about
print(
    f"time to produce one 4-node training step ({TRAIN_SAMPLES_PER_STEP} samples): "
    f"{TRAIN_SAMPLES_PER_STEP / (total_ranks / mean):.1f} s"
)

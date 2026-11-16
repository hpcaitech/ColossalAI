"""Per-window cost of one dataset read, to find the windows that cost tens of seconds.

The training step stalls ~170-200 s at the same update index on every node and every run, while the surrounding
updates take 31 s. During the stall the rank sits inside ``pack_batch`` and its DataLoader workers burn CPU, so the
suspicion is a handful of windows whose ``_read_frames`` decodes far more than the 33 frames it keeps -- a timestamp
seek that lands before a long keyframe interval makes the decoder walk forward through the file.

    python window_cost.py [first] [count]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.worldmodel.cosmos3.droid_dataset import DroidActionPolicyDataset

DATA = "/mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success"

first = int(sys.argv[1]) if len(sys.argv) > 1 else 0
count = int(sys.argv[2]) if len(sys.argv) > 2 else 120

dataset = DroidActionPolicyDataset(DATA, chunk_length=32, fps=15.0, resolution="480", augment=False,
                                   max_episodes=64)
print(f"windows={len(dataset)} scanning [{first}, {first + count})", flush=True)

slow = []
started = time.perf_counter()
for index in range(first, min(first + count, len(dataset))):
    window = dataset.windows[index]
    t0 = time.perf_counter()
    dataset[index]
    elapsed = time.perf_counter() - t0
    if elapsed > 3.0:
        slow.append((index, elapsed))
        print(f"slow index={index} episode={window.episode_index} start={window.start} {elapsed:.1f}s", flush=True)

total = time.perf_counter() - started
print(f"scanned={min(count, len(dataset) - first)} total={total:.1f}s mean={total / count:.2f}s "
      f"slow_windows={len(slow)}")
if slow:
    worst = max(slow, key=lambda item: item[1])
    print(f"worst index={worst[0]} {worst[1]:.1f}s")

"""Split one sample's CPU cost into decode, geometry and parquet, to place the periodic stall.

Measured on node6: one camera's ``_read_frames`` seeks and decodes exactly 33-34 frames in 0.04 s, so the timestamp
seek is not the cost -- whatever makes a sample take 1.4 s, and whatever makes a whole rank stall for 170 s, is
elsewhere in ``__getitem__``. This times the three parts separately.

    python sample_cost.py [samples]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.worldmodel.cosmos3.droid_dataset import CAMS, DroidActionPolicyDataset, compose_concat_view

DATA = "/mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success"

count = int(sys.argv[1]) if len(sys.argv) > 1 else 5
dataset = DroidActionPolicyDataset(DATA, chunk_length=32, fps=15.0, resolution="480", augment=False,
                                   max_episodes=64)
print(f"windows={len(dataset)}", flush=True)

decode = geometry = parquet = full = 0.0
for index in range(count):
    window = dataset.windows[index]

    t0 = time.perf_counter()
    views = {camera: dataset._read_frames(camera, window).float() / 255.0 for camera in CAMS}
    t1 = time.perf_counter()
    composite = compose_concat_view(views["wrist"], views["left"], views["right"])
    quantised = (composite * 255.0).clamp(0.0, 255.0).to(torch.uint8)
    t2 = time.perf_counter()
    action, state = dataset._read_actions(window)
    t3 = time.perf_counter()
    dataset[index]
    t4 = time.perf_counter()

    decode += t1 - t0
    geometry += t2 - t1
    parquet += t3 - t2
    full += t4 - t3
    print(f"index={index} decode={1e3 * (t1 - t0):6.1f}ms compose_cast={1e3 * (t2 - t1):6.1f}ms "
          f"parquet={1e3 * (t3 - t2):6.1f}ms | full_getitem={1e3 * (t4 - t3):7.1f}ms "
          f"(3 cameras x 33 frames, {tuple(quantised.shape)})", flush=True)

print(f"mean decode={1e3 * decode / count:.1f}ms compose={1e3 * geometry / count:.1f}ms "
      f"parquet={1e3 * parquet / count:.1f}ms full={1e3 * full / count:.1f}ms")

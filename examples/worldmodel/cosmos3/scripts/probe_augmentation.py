"""Check that ``DroidActionPolicyDataset(augment=True)`` really augments, and what it costs.

The reference bench in ``/mnt/disk0/jiangzs/share/bench_augmentation.py`` times synthetic tensors and never
touches the dataset, so it cannot tell whether the dataset's augmentation path works -- a silent no-op would
benchmark exactly the same. This compares the two dataset modes on the same window:

* ``augment=True`` vs ``False`` must differ (the transforms are applied at all);
* two draws with ``augment=True`` must differ (they are random, not a fixed crop);
* canvas and content geometry must be identical in both cases (augmentation moves pixels, not the layout).

    python probe_augmentation.py [samples] [max_episodes]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.worldmodel.cosmos3 import droid_dataset
from examples.worldmodel.cosmos3.droid_dataset import DroidActionPolicyDataset

DATA = "/mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success"


def relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    return ((left - right).pow(2).sum().sqrt() / right.pow(2).sum().sqrt()).item()


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    max_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    print(f"torch {torch.__version__}  torchvision {droid_dataset.T.__name__ if droid_dataset.T else 'MISSING'}"
          f"  import error: {droid_dataset.TORCHVISION_IMPORT_ERROR}")
    print(f"threads {torch.get_num_threads()}")

    common = dict(chunk_length=32, fps=15.0, resolution="480", max_episodes=max_episodes)
    plain = DroidActionPolicyDataset(DATA, augment=False, **common)
    aug = DroidActionPolicyDataset(DATA, augment=True, **common)
    print(f"windows {len(plain)}")

    for index in range(count):
        started = time.perf_counter()
        base = plain[index]
        plain_ms = (time.perf_counter() - started) * 1e3
        started = time.perf_counter()
        once = aug[index]
        aug_ms = (time.perf_counter() - started) * 1e3
        twice = aug[index]

        same_geometry = torch.equal(base["canvas_hw"], once["canvas_hw"]) and torch.equal(
            base["content_hw"], once["content_hw"]
        )
        print(
            f"window {index}: plain {plain_ms:7.1f} ms  augment {aug_ms:7.1f} ms  "
            f"shape {tuple(once['video'].shape)}  "
            f"aug-vs-plain relL2 {relative_l2(once['video'], base['video']):.4f}  "
            f"draw-to-draw relL2 {relative_l2(twice['video'], once['video']):.4f}  "
            f"geometry identical {same_geometry}  "
            f"range [{once['video'].min():.2f}, {once['video'].max():.2f}]"
        )

    views = {cam: torch.rand(33, 3, 360, 640) for cam in ("wrist", "left", "right")}
    aug._augment(views)
    repeats = 5
    started = time.perf_counter()
    for _ in range(repeats):
        aug._augment(views)
    print(f"_augment on 99x3x360x640: {(time.perf_counter() - started) / repeats * 1e3:.1f} ms")


if __name__ == "__main__":
    main()

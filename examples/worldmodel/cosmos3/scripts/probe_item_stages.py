"""Where do the seconds in ``DroidActionPolicyDataset.__getitem__`` actually go?

The dataloader is what the steady step waits on, and ``probe_augmentation.py`` only separates "plain" from
"augmented". This replays ``__getitem__`` stage by stage on real windows, with the same arguments the trainer uses
(action cache on, augmentation on, content pad 16), and prints each stage's share.

    python scripts/probe_item_stages.py [windows]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from examples.worldmodel.cosmos3.droid_dataset import (
    CAMS,
    DroidActionPolicyDataset,
    closest_canvas,
    compose_concat_view,
    resize_and_reflect_pad,
)

DATA = "/mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success"
CACHE = "/mnt/disk0/jiangzs/nzf/action_cache"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=480)
    parser.add_argument("--pad", type=int, default=16)
    args = parser.parse_args()

    dataset = DroidActionPolicyDataset(
        DATA,
        chunk_length=32,
        fps=15,
        resolution=str(args.resolution),
        augment=True,
        max_episodes=64,
        action_cache=Path(CACHE),
        video_content_pad=args.pad,
    )
    print(f"[item] {len(dataset)} windows, augment={dataset.augment}, pad={args.pad}")

    totals = {}
    for index in range(min(args.windows, len(dataset))):
        win = dataset.windows[index]
        marks = {}

        started = time.perf_counter()
        views = {cam: dataset._read_frames(cam, win).float() / 255.0 for cam in CAMS}
        marks["read_frames"] = time.perf_counter() - started

        started = time.perf_counter()
        views = dataset._augment(views)
        marks["augment"] = time.perf_counter() - started

        started = time.perf_counter()
        comp = compose_concat_view(views["wrist"], views["left"], views["right"])
        marks["compose"] = time.perf_counter() - started

        started = time.perf_counter()
        th, tw, _aspect = closest_canvas(*comp.shape[-2:], resolution=args.resolution)
        marks["canvas"] = time.perf_counter() - started

        started = time.perf_counter()
        comp8 = (comp * 255.0).clamp(0.0, 255.0).to(torch.uint8)
        marks["quantise"] = time.perf_counter() - started

        started = time.perf_counter()
        video, _content = resize_and_reflect_pad(comp8.float(), th, tw, pad_columns=args.pad)
        marks["resize_pad"] = time.perf_counter() - started

        started = time.perf_counter()
        video = video / 127.5 - 1.0
        marks["normalise"] = time.perf_counter() - started

        started = time.perf_counter()
        dataset._read_actions(win)
        marks["actions"] = time.perf_counter() - started

        for name, seconds in marks.items():
            totals[name] = totals.get(name, 0.0) + seconds
        if index == 0:
            print(f"[item] composite {tuple(comp.shape)} -> canvas {th}x{tw}, video {tuple(video.shape)}")
        print(
            "  window %d: %s  total %.2f s"
            % (
                index,
                "  ".join(f"{name} {seconds * 1e3:.0f}ms" for name, seconds in marks.items()),
                sum(marks.values()),
            ),
            flush=True,
        )

    print("\n[item] average per sample")
    for name, seconds in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<12}{seconds / args.windows * 1e3:8.0f} ms  " f"({100 * seconds / sum(totals.values()):4.1f}%)")


if __name__ == "__main__":
    main()

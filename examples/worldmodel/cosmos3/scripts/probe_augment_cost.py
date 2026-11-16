"""What does the augmentation chain cost per sample, and which transform owns it?

``droid_dataset._augment`` mirrors upstream: one ``T.Compose`` call on ``cat([wrist, left, right])`` with
``RandomCrop(0.95*hw) -> Resize(hw, antialias=True) -> ColorJitter(0.3, 0.4, 0.5, 0.08)``. That runs on 3 cameras x
33 frames of 540x640 float32 -- 411 MB per sample -- on the *worker* CPUs, and the dataloader is what the steady step
waits on. This times the chain, each stage, and each of ColorJitter's four adjustments on a synthetic tensor of
exactly that shape.

    python scripts/probe_augment_cost.py [threads]
"""

from __future__ import annotations

import argparse
import time

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF


def bench(fn, iters=3):
    fn()
    started = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - started) / iters * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cameras", type=int, default=3)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    shape = (args.cameras * args.frames, 3, args.height, args.width)
    x = torch.rand(shape)
    gib = x.numel() * 4 / 2**30

    chain = T.Compose(
        [
            T.RandomCrop((int(args.height * 0.95), int(args.width * 0.95))),
            T.Resize((args.height, args.width), antialias=True),
            T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
        ]
    )
    stages = {
        "compose": lambda: chain(x),
        "brightness": lambda: TF.adjust_brightness(x, 1.15),
        "contrast": lambda: TF.adjust_contrast(x, 1.2),
        "saturation": lambda: TF.adjust_saturation(x, 1.25),
        "hue": lambda: TF.adjust_hue(x, 0.04),
    }

    print(f"[aug] input {tuple(shape)} = {gib:.2f} GiB, {args.threads} torch thread(s)")
    for name, fn in stages.items():
        ms = bench(fn)
        print(f"[aug] {name:<11}{ms:9.1f} ms   ({2 * gib * 1000 / ms:6.2f} GiB/s of one read+write pass)")


if __name__ == "__main__":
    main()

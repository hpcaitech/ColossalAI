"""Is ``FastColorJitter`` bit-identical to ``torchvision.ColorJitter``, and how much faster?

The two draw their parameters from the same generator in the same order (``make_params`` is torchvision's own), so
seeding before each call and comparing element-wise covers both the draws and the arithmetic. Then it times both on a
production-shaped tensor.

    python scripts/probe_fast_jitter.py [threads]
"""

from __future__ import annotations

import argparse
import time

import torch
import torchvision.transforms.v2 as T


def bench(fn, iters=3):
    fn()
    started = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - started) / iters * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--frames", type=int, default=99)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()

    from examples.worldmodel.cosmos3.fast_color_jitter import FastColorJitter

    torch.set_num_threads(args.threads)
    x = torch.rand(args.frames, 3, args.height, args.width)

    reference = T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08)
    fast = FastColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08)

    draws_match = True
    worst = 0
    for seed in range(3):
        torch.manual_seed(seed)
        expected = reference(x)
        torch.manual_seed(seed)
        actual = fast(x)
        if not torch.equal(actual, expected):
            draws_match = False
            worst = max(worst, int((actual != expected).sum()))
    print(f"[jitter] three seeds: bit_equal={draws_match} worst mismatched {worst}/{x.numel()}")

    ms_reference = bench(lambda: reference(x))
    ms_fast = bench(lambda: fast(x))
    print(
        f"[jitter] {args.threads} threads: ColorJitter {ms_reference:.0f} ms | FastColorJitter "
        f"{ms_fast:.0f} ms | {100 * (1 - ms_fast / ms_reference):+.0f}%"
    )
    print(
        "[jitter] per sample saved: "
        f"{(ms_reference - ms_fast) / 1000:.2f} s; 32 samples/step / 4 workers -> "
        f"{32 * (ms_reference - ms_fast) / 1000 / 4:.1f} s per worker per step"
    )


if __name__ == "__main__":
    main()

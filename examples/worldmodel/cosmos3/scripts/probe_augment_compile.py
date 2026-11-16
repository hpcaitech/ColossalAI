"""Can ``torch.compile`` make the hue adjustment cheap without changing a single bit?

``adjust_hue`` is 71% of the augmentation chain (1.63 s of 2.29 s per sample at 4 threads) because the HSV round trip
is a long chain of full-tensor temporaries. The workers run on CPU, so ``torch.compile`` (inductor, CPU) can fuse that
chain. Inductor keeps fp32 semantics per op, so the output should be bit-identical -- this measures both.

    python scripts/probe_augment_compile.py [threads]
"""

from __future__ import annotations

import argparse
import time

import torch
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
    parser.add_argument("--frames", type=int, default=99)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--factor", type=float, default=0.04)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    x = torch.rand(args.frames, 3, args.height, args.width)

    reference = TF.adjust_hue(x, args.factor)
    ms_eager = bench(lambda: TF.adjust_hue(x, args.factor))
    print(f"[hue] eager       {ms_eager:8.1f} ms")

    try:
        compiled = torch.compile(lambda t: TF.adjust_hue(t, args.factor), dynamic=False)
        out = compiled(x)
    except Exception as error:  # inductor unavailable / unsupported op
        print(f"[hue] torch.compile failed: {type(error).__name__}: {str(error)[:120]}")
        return

    equal = torch.equal(out, reference)
    mismatched = int((out != reference).sum())
    print(f"[hue] compiled bit_equal={equal} mismatched {mismatched}/{reference.numel()}")
    ms_compiled = bench(lambda: compiled(x))
    print(f"[hue] compiled    {ms_compiled:8.1f} ms  ({100 * (1 - ms_compiled / ms_eager):+.1f}%)")


if __name__ == "__main__":
    main()

"""One-pass versus two-pass fused RMSNorm on the widest shapes.

The one-pass kernel keeps a [C, chunk] tile in UB, which caps the row width at chunk*2 bytes (1 KB at C=160) and
leaves it at ~356 GB/s of effective traffic. The two-pass variant (reduce into a per-column norm, then apply)
uses a single row of 16 KB, at the cost of reading x twice. Same arithmetic, same order, so the gate is bit
equality; the question is only speed.

    python scripts/probe_norm_twopass.py --checkpoint /mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID
"""

from __future__ import annotations

import argparse

import torch

SHAPES = ((8, 160, 356864), (8, 320, 89216), (8, 640, 11152), (8, 160, 22304))


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    import examples.worldmodel.cosmos3.vae_norm_kernel as nk

    device = torch.device("npu:0")
    for b, c, s in SHAPES:
        torch.manual_seed(0)
        x = torch.randn(b, c, s, device=device, dtype=torch.bfloat16)
        gamma = torch.rand(c, device=device, dtype=torch.float32) * 0.5 + 0.5
        scale = float(c) ** 0.5

        one = nk._kernel(b, c, s, scale)
        y_one = one(x, gamma)

        plan = nk._plan_two_pass(c, s)
        if plan is None:
            print(f"  {b}x{c}x{s}: no two-pass plan")
            continue
        chunk, nblk = plan
        reduce_k = nk._channel_rmsnorm_reduce(b, c, s, chunk, nblk)
        apply_k = nk._channel_rmsnorm_apply(b, c, s, chunk, nblk, scale)
        norm = torch.empty(b, s, device=device, dtype=torch.float32)
        out = torch.empty_like(x)

        def two():
            reduce_k(x, norm)
            apply_k(x, gamma, norm, out)

        two()
        equal = torch.equal(out, y_one)
        diff = (out.float() - y_one.float()).norm() / y_one.float().norm().clamp_min(1e-12)
        ms_one = bench(lambda: one(x, gamma), args.iters)
        ms_two = bench(two, args.iters)
        gbps_one = 2 * x.numel() * 2 / (ms_one * 1e6)
        gbps_two = 3 * x.numel() * 2 / (ms_two * 1e6)
        print(f"  {b}x{c}x{s}: one {ms_one:.3f} ms ({gbps_one:.0f} GB/s) | two {ms_two:.3f} ms "
              f"({gbps_two:.0f} GB/s) | bit_equal={equal} relL2={float(diff):.2e}", flush=True)
        del x, y_one, out, norm
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()

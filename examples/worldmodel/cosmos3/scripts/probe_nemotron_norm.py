"""Can a fused kernel reproduce ``Cosmos3NemotronRMSNorm`` bit-exactly?

The DiT's norm is not the fused NPU one. ``Cosmos3NemotronRMSNorm.forward`` upcasts to fp32 and then runs
``pow(2) -> mean(-1) -> * rsqrt(v+eps) -> * weight.float() -> to(bf16)`` as five separate kernels, four times per
layer. All of those are exact except the reduction: a fused kernel can reproduce every elementwise step, but only if
it sums in the same order as ``mean(-1)``.

This probe answers that first question on real shapes and random data:

    python scripts/probe_nemotron_norm.py

The candidates are a plain left-to-right sum (``cumsum[..., -1]``), a pairwise tree, and an 8-lane block sum
followed by each of the two. Whichever matches exactly is what a TileLang kernel has to implement.
"""

from __future__ import annotations

import argparse

import torch


def bench(fn, iters=50):
    for _ in range(5):
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


def sequential(x):
    return torch.cumsum(x, dim=-1)[..., -1]


def tree(x):
    while x.shape[-1] > 1:
        x = x[..., 0::2] + x[..., 1::2]
    return x[..., 0]


def lanes_then(x, lanes, combine):
    """Each lane sums a strided subsequence (what a vector load does), then the lanes are folded."""
    rows = x.reshape(*x.shape[:-1], x.shape[-1] // lanes, lanes)
    partial = torch.cumsum(rows, dim=-2)[..., -1, :]  # running sum per lane over its own subsequence
    return combine(partial)


def blocks_then(x, blocks, combine_inner, combine_outer):
    """Split the row into contiguous blocks, sum each, then fold the block sums."""
    rows = x.reshape(*x.shape[:-1], blocks, x.shape[-1] // blocks)
    partial = combine_inner(rows)  # reduce the contiguous in-block axis
    return combine_outer(partial)  # then fold the block axis


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("npu:0")
    torch.manual_seed(0)
    x = torch.randn(args.rows, args.hidden, device=device, dtype=torch.bfloat16)
    x32 = x.float()
    squares = x32 * x32

    reference = squares.mean(-1)
    total = squares.sum(-1)
    print(
        f"[mean] shapes {tuple(x.shape)}; mean == sum/{args.hidden}: " f"{torch.equal(reference, total / args.hidden)}"
    )

    candidates = [("sequential", sequential), ("tree", tree)]
    for lanes in (2, 4, 8, 16, 32, 64):
        if args.hidden % lanes:
            continue
        candidates.append((f"{lanes}lane+seq", lambda t, n=lanes: lanes_then(t, n, sequential)))
        candidates.append((f"{lanes}lane+tree", lambda t, n=lanes: lanes_then(t, n, tree)))
    for blocks in (2, 4, 8, 16, 32):
        if args.hidden % blocks:
            continue
        candidates.append((f"{blocks}blk+seq+seq", lambda t, n=blocks: blocks_then(t, n, sequential, sequential)))
        candidates.append((f"{blocks}blk+tree+tree", lambda t, n=blocks: blocks_then(t, n, tree, tree)))

    exact = []
    for name, candidate in candidates:
        got = candidate(squares)
        mismatched = int((got != total).sum())
        worst = float((got - total).abs().max())
        state = "EXACT" if mismatched == 0 else f"{mismatched}/{args.rows}"
        print(f"[sum ] {name:<15} mismatched rows {state}  max|diff| {worst:.3e}")
        if mismatched == 0:
            exact.append(name)
    print(f"[sum ] exact matches: {exact or 'none'}")

    # The other order-sensitive primitive: does torch.rsqrt equal a precise 1/sqrt on the actual variance values?
    # A fused kernel can only reproduce it if the answer is yes (then T.tile.sqrt + division matches) or if it uses
    # the same approximation instruction.
    with torch.no_grad():
        variance = squares.mean(-1, keepdim=True) + 1e-5
        rsqrt_torch = torch.rsqrt(variance)
        precise = 1.0 / torch.sqrt(variance)
        print(
            f"[rsqrt] torch.rsqrt == 1/sqrt: {torch.equal(rsqrt_torch, precise)} "
            f"(mismatched {int((rsqrt_torch != precise).sum())}/{variance.numel()})"
        )

    # Cost of the reference chain versus the traffic a fused kernel would move.
    weight = torch.rand(args.hidden, device=device, dtype=torch.bfloat16)
    eps = 1e-5

    def reference_chain():
        h = x.float()
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + eps)
        return (weight.float() * h).to(x.dtype)

    def fused_floor():
        # Same values, one pass: read bf16, accumulate fp32 squares, write bf16. This is the bandwidth floor a
        # TileLang kernel can reach; it is not the reference result, only the cost of moving the data once.
        variance = (x.float() * x.float()).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)

    ms_ref = bench(reference_chain, args.iters)
    ms_floor = bench(fused_floor, args.iters)
    moved_ref = args.rows * args.hidden * (2 + 4 + 8 + 8 + 8 + 8 + 6)
    print(
        f"[time] reference chain {ms_ref:.3f} ms for {args.rows}x{args.hidden} "
        f"({moved_ref / (ms_ref * 1e6):.0f} GB/s of the ~40 bytes/element it touches)"
    )
    print(f"[time] torch floor (cast+mean+rsqrt+mul, 4 kernels) {ms_floor:.3f} ms")
    print("[note] 4 norms per layer x 28 layers x (fwd + ~2x bwd) at this row count")


if __name__ == "__main__":
    main()

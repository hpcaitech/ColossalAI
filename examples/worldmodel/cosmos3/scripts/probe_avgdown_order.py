"""Can AvgDown3D's ``permute+contiguous+mean`` be replaced by a fused reduction, bit for bit?

The permute materialises a full copy of the activation before the mean (0.44 s/step across the encoder). A fused
kernel would read the same elements and average them directly, so the question that decides whether it is worth
writing is: does the arithmetic agree exactly?

diffusers reduces ``mean(dim=2)`` of a ``[b, out_c, group_size, t', h', w']`` view, where the group axis holds
``group_size = in_c*ft*fs*fs/out_c`` consecutive entries of the merged ``(c, t, h, w)`` axis - 8, 4 and 2 for the
encoder's three real configurations. This probe builds a random bf16 activation of each production shape, runs the
reference chain, and compares four fused formulations:

* ``seq``     - float32 accumulation over the group in memory order, one division, one rounding;
* ``pair``    - the same values summed as a pairwise float32 tree;
* ``bf16``    - bf16 accumulation (what a naive kernel would do);
* ``sum_div`` - float32 sum then multiply by 1/group instead of dividing.

``--iters`` repeats with fresh data, because one draw can agree by luck.

    python scripts/probe_avgdown_order.py --iters 5
"""

from __future__ import annotations

import argparse

import torch


def merged_view(x: torch.Tensor, ft: int, fs: int, out_c: int) -> torch.Tensor:
    """[b, in_c, t, h, w] -> [b, out_c, group, t', h', w'], the tensor diffusers averages over dim=2."""
    b, c, t, h, w = x.shape
    group = ft * fs * fs
    view = x.view(b, c, t // ft, ft, h // fs, fs, w // fs, fs)
    terms = [view[:, :, :, a, :, hh, :, ww] for a in range(ft) for hh in range(fs) for ww in range(fs)]
    stacked = torch.stack(terms, dim=2)                       # [b, c, t', h', w', group]
    merged = stacked.permute(0, 1, 5, 2, 3, 4).reshape(b, c * group, t // ft, h // fs, w // fs)
    group_size = c * group // out_c
    return merged.view(b, out_c, group_size, t // ft, h // fs, w // fs)


def reduce_group(x: torch.Tensor, how: str) -> torch.Tensor:
    g = x.shape[2]
    if how == "bf16":
        total = x[:, :, 0]
        for index in range(1, g):
            total = total + x[:, :, index]
        return (total / g).to(torch.bfloat16)
    wide = x.float()
    if how == "pair":
        acc = wide
        while acc.shape[2] > 1:
            acc = acc[:, :, 0::2] + acc[:, :, 1::2]
        total = acc[:, :, 0]
    else:
        total = wide[:, :, 0]
        for index in range(1, g):
            total = total + wide[:, :, index]
    if how == "sum_div":
        return (total * (1.0 / g)).to(torch.bfloat16)
    return (total / g).to(torch.bfloat16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--shape", default="8,160,8,64,80", help="b,c,t,h,w of the activation")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = parser.parse_args()

    device = torch.device("npu:0")
    dtype = getattr(torch, args.dtype)
    b, c, t, h, w = (int(value) for value in args.shape.split(","))
    # The encoder's three AvgDown configurations (in_c, out_c, factor_t, factor_s).
    configs = [(c, c, 2, 2), (c, 2 * c, 2, 2), (2 * c, 4 * c, 1, 2)]
    orders = ["seq", "pair", "bf16", "sum_div"]

    for in_c, out_c, ft, fs in configs:
        group_size = in_c * ft * fs * fs // out_c
        stats = {order: {"equal": 0, "bad": 0, "worst": 0.0} for order in orders}
        elements = 0
        for _ in range(args.iters):
            x = torch.randn(b, in_c, t, h, w, device=device, dtype=torch.float32).to(dtype)
            view = merged_view(x, ft, fs, out_c)
            ref = view.mean(dim=2)
            elements += ref.numel()
            for order in orders:
                got = reduce_group(view, order)
                if torch.equal(got, ref):
                    stats[order]["equal"] += 1
                    continue
                stats[order]["bad"] += int((got != ref).sum().item())
                stats[order]["worst"] = max(stats[order]["worst"],
                                            float((got.float() - ref.float()).abs().max().item()))
        print(f"[avgdown] {in_c}->{out_c} ft={ft} fs={fs} group_size={group_size} shape={(b, in_c, t, h, w)}")
        for order in orders:
            item = stats[order]
            verdict = "BIT-IDENTICAL" if item["equal"] == args.iters else f"differs on {item['bad']}/{elements}"
            print(f"    {order:<8} {item['equal']}/{args.iters} runs equal   {verdict:<28} worst |d| {item['worst']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

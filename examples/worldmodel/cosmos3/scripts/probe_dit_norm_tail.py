"""Is the fused DiT-norm tail bit-identical, forward and backward?

``dit_norm_tail.py`` keeps the reference head (cast, pow, mean, rsqrt) and fuses only ``h * r``, ``w * u`` and the
bf16 cast into one kernel. Everything involved is elementwise, so bit equality is the gate -- for the output *and*
for both gradients.

    python scripts/probe_dit_norm_tail.py [rows]
"""

from __future__ import annotations

import argparse

import torch


def bench(fn, iters=20):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=49152)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    from diffusers.models.transformers.transformer_cosmos3 import Cosmos3NemotronRMSNorm

    import examples.worldmodel.cosmos3.dit_norm_tail as dt

    device = torch.device("npu:0")
    module = Cosmos3NemotronRMSNorm(args.hidden, eps=1e-5).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        module.weight.copy_(torch.randn(args.hidden, device=device, dtype=torch.float32).clamp(0.5, 1.5))

    torch.manual_seed(0)
    x = torch.randn(args.rows, args.hidden, device=device, dtype=torch.bfloat16) * 0.5
    with torch.no_grad():
        reference = dt._reference_forward(module, x)
        fused = dt._tail_forward(module, x)
    print(f"[fwd ] bit_equal={torch.equal(fused, reference)} mismatched {int((fused != reference).sum())}/"
          f"{reference.numel()}")

    torch.manual_seed(7)
    seed_grad = torch.randn(args.rows, args.hidden, device=device, dtype=torch.bfloat16) * 0.05
    xr = (torch.randn(args.rows, args.hidden, device=device, dtype=torch.bfloat16) * 0.5).requires_grad_(True)
    module.weight.grad = None
    dt._reference_forward(module, xr).backward(seed_grad)
    ref_gx, ref_gw = xr.grad.clone(), module.weight.grad.clone()

    xr.grad = None
    module.weight.grad = None
    dt._tail_forward(module, xr).backward(seed_grad)
    gx, gw = xr.grad, module.weight.grad
    print(f"[bwd ] x grad bit_equal={torch.equal(gx, ref_gx)} mismatched {int((gx != ref_gx).sum())}/{gx.numel()}")
    print(f"[bwd ] w grad bit_equal={torch.equal(gw, ref_gw)} mismatched {int((gw != ref_gw).sum())}/{gw.numel()}")

    with torch.no_grad():
        ms_ref = bench(lambda: dt._reference_forward(module, x), args.iters)
        ms_fused = bench(lambda: dt._tail_forward(module, x), args.iters)
    print(f"[time] {args.rows}x{args.hidden}: reference {ms_ref:.3f} ms | tail-fused {ms_fused:.3f} ms "
          f"({100 * (1 - ms_fused / ms_ref):+.1f}%)  -> about {(ms_ref - ms_fused) * 394 / 1000:.2f} s/step")


if __name__ == "__main__":
    main()

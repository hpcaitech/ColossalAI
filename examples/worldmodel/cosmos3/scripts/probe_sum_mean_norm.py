"""Is `sum(-1)/H` bit-identical to `mean(-1)` in the DiT norm -- output and both gradients?

``dit_norm_kernel.enable_sum_mean_norm`` rewrites ``variance = h.pow(2).mean(-1)`` as ``h.pow(2).sum(-1) / H``. H is
2048, a power of two, so the division is exact and the values should be identical; the point is that autograd's
mean-backward divides the *whole* [tokens, 2048] gradient, while the sum-backward only divides a [tokens, 1] vector.
This checks the values, then reports what the rewrite costs and saves at production shape.

    python scripts/probe_sum_mean_norm.py [rows]
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

    import examples.worldmodel.cosmos3.dit_norm_kernel as dk

    device = torch.device("npu:0")
    module = Cosmos3NemotronRMSNorm(args.hidden, eps=1e-5).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        module.weight.copy_(torch.randn(args.hidden, device=device, dtype=torch.float32).clamp(0.5, 1.5))

    torch.manual_seed(0)
    x = torch.randn(args.rows, args.hidden, device=device, dtype=torch.bfloat16) * 0.5
    with torch.no_grad():
        reference = dk._reference_forward(module, x)
        candidate = dk._sum_mean_forward(module, x)
    print(f"[fwd ] bit_equal={torch.equal(candidate, reference)} "
          f"mismatched {int((candidate != reference).sum())}/{reference.numel()}")

    torch.manual_seed(7)
    seed_grad = torch.randn(args.rows, args.hidden, device=device, dtype=torch.bfloat16) * 0.05
    xr = (torch.randn(args.rows, args.hidden, device=device, dtype=torch.bfloat16) * 0.5).requires_grad_(True)
    module.weight.grad = None
    dk._reference_forward(module, xr).backward(seed_grad)
    ref_gx, ref_gw = xr.grad.clone(), module.weight.grad.clone()

    xr.grad = None
    module.weight.grad = None
    dk._sum_mean_forward(module, xr).backward(seed_grad)
    print(f"[bwd ] x grad bit_equal={torch.equal(xr.grad, ref_gx)} mismatched {int((xr.grad != ref_gx).sum())}/"
          f"{xr.grad.numel()} | w grad bit_equal={torch.equal(module.weight.grad, ref_gw)}")

    def fwd_bwd(forward):
        def run():
            leaf = x.detach().requires_grad_(True)
            module.weight.grad = None
            forward(module, leaf).backward(seed_grad)
        return run

    ms_ref = bench(fwd_bwd(dk._reference_forward), args.iters)
    ms_sum = bench(fwd_bwd(dk._sum_mean_forward), args.iters)
    print(f"[time] {args.rows}x{args.hidden} fwd+bwd: mean {ms_ref:.3f} ms | sum/H {ms_sum:.3f} ms "
          f"({100 * (1 - ms_sum / ms_ref):+.1f}%) -> {(ms_ref - ms_sum) * 394 / 1000:+.2f} s/step at 394 chains")


if __name__ == "__main__":
    main()

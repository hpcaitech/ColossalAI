"""Is the fused DiT norm exact, and how much does it save?

``Cosmos3NemotronRMSNorm`` runs as five fp32 kernels (see ``dit_norm_kernel.py``); the fused replacement is the
vendor ``npu_rms_norm`` wrapped in a ``Function`` whose backward repeats the reference chain in torch ops. Three
questions, one run:

1. how far the vendor forward is from the decomposed chain (bit count + relL2),
2. whether the custom backward is bit-identical to autograd's,
3. what one call costs at production row counts (the 8k-row case is L2-resident and not representative).

    python scripts/probe_dit_norm.py
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
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--timing-rows", type=int, default=49152)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    import torch_npu
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
        npu_out = torch_npu.npu_rms_norm(x, module.weight, epsilon=module.eps)[0]
    mismatched = int((npu_out != reference).sum())
    rel = float((npu_out.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12))
    print(f"[fwd  ] {args.rows}x{args.hidden}: npu_rms_norm mismatched {mismatched}/{reference.numel()} "
          f"({100 * mismatched / reference.numel():.5f}%) relL2={rel:.2e}")

    # Backward: the Function's chain against autograd through the reference forward.
    torch.manual_seed(7)
    seed_grad = torch.randn(args.rows, args.hidden, device=device, dtype=torch.bfloat16) * 0.05
    xr = (torch.randn(args.rows, args.hidden, device=device, dtype=torch.bfloat16) * 0.5).requires_grad_(True)
    module.weight.grad = None
    dk._reference_forward(module, xr).backward(seed_grad)
    ref_gx, ref_gw = xr.grad.clone(), module.weight.grad.clone()

    xr.grad = None
    module.weight.grad = None
    out = dk._FusedNemotronNorm.apply(xr, module.weight, module.eps)
    out.backward(seed_grad)
    gx, gw = xr.grad, module.weight.grad
    print(f"[bwd  ] x grad bit_equal={torch.equal(gx, ref_gx)} mismatched {int((gx != ref_gx).sum())}/{gx.numel()} "
          f"relL2={float((gx.float() - ref_gx.float()).norm() / ref_gx.float().norm().clamp_min(1e-12)):.2e}")
    print(f"[bwd  ] w grad bit_equal={torch.equal(gw, ref_gw)} mismatched {int((gw != ref_gw).sum())}/{gw.numel()} "
          f"relL2={float((gw.float() - ref_gw.float()).norm() / ref_gw.float().norm().clamp_min(1e-12)):.2e}")

    with torch.no_grad():
        big = torch.randn(args.timing_rows, args.hidden, device=device, dtype=torch.bfloat16) * 0.5
        ms_ref = bench(lambda: dk._reference_forward(module, big), args.iters)
        ms_npu = bench(lambda: torch_npu.npu_rms_norm(big, module.weight, epsilon=module.eps)[0], args.iters)
    print(f"[time] {args.timing_rows}x{args.hidden}: reference chain {ms_ref:.3f} ms | npu_rms_norm {ms_npu:.3f} ms "
          f"| {100 * (1 - ms_npu / ms_ref):+.1f}%")
    print(f"[time] 394 chains per step -> about {(ms_ref - ms_npu) * 394 / 1000:.2f} s/step saved")


if __name__ == "__main__":
    main()

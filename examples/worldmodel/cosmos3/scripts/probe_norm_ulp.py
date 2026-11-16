"""If the DiT norm's row sum were replaced by a reproducible one, how many bf16 outputs would move?

``probe_nemotron_norm.py`` shows no candidate reduction matches ``mean(-1)`` bitwise: the best is off by one fp32
ulp on 35% of rows. That is an *intermediate* difference. What the training sees is the bf16 output, and a 1-ulp
fp32 change in the variance only reaches it through ``rsqrt`` and two multiplies -- usually without flipping the
final rounding.

This measures the end-to-end number on production-shaped data, for the two cheapest candidate reductions and for
the vendor's fused op, so the trade-off is a measured rate rather than a guess.

    python scripts/probe_norm_ulp.py
"""

from __future__ import annotations

import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=49152)
    parser.add_argument("--hidden", type=int, default=2048)
    args = parser.parse_args()

    import torch_npu
    from diffusers.models.transformers.transformer_cosmos3 import Cosmos3NemotronRMSNorm

    device = torch.device("npu:0")
    eps = 1e-5
    module = Cosmos3NemotronRMSNorm(args.hidden, eps=eps).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        module.weight.copy_(torch.randn(args.hidden, device=device, dtype=torch.float32).clamp(0.5, 1.5))
        weight = module.weight.float()

        torch.manual_seed(0)
        x = torch.randn(args.rows, args.hidden, device=device, dtype=torch.bfloat16) * 0.5
        h = x.float()
        squares = h * h

        reference = (weight * (h * torch.rsqrt(squares.mean(-1, keepdim=True) + eps))).to(x.dtype)

        candidates = {
            "sequential": torch.cumsum(squares, dim=-1)[..., -1:] / args.hidden,
            "64lane+tree": None,
        }
        lanes = 64
        partial = torch.cumsum(squares.reshape(args.rows, args.hidden // lanes, lanes), dim=1)[:, -1, :]
        tree = partial
        while tree.shape[-1] > 1:
            tree = tree[..., 0::2] + tree[..., 1::2]
        candidates["64lane+tree"] = tree / args.hidden

        for name, variance in candidates.items():
            r = torch.rsqrt(variance + eps)
            out = (weight * (h * r)).to(x.dtype)
            mismatched = int((out != reference).sum())
            rel = float((out.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12))
            print(f"[ulp  ] {name:<12} bf16 mismatched {mismatched}/{reference.numel()} "
                  f"({100 * mismatched / reference.numel():.4f}%) relL2={rel:.2e}")

        npu = torch_npu.npu_rms_norm(x, module.weight, epsilon=eps)[0]
        mismatched = int((npu != reference).sum())
        rel = float((npu.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12))
        print(f"[ulp  ] npu_rms_norm  bf16 mismatched {mismatched}/{reference.numel()} "
              f"({100 * mismatched / reference.numel():.4f}%) relL2={rel:.2e}")


if __name__ == "__main__":
    main()

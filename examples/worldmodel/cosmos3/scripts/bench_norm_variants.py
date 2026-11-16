"""Compare RMSNorm implementations on the shapes the Cosmos 3 decoder actually uses.

The decoder's ``Cosmos3NemotronRMSNorm`` is a five-op fp32 chain (cast, pow, mean, rsqrt, mul, cast). Ascend
exposes ``torch_npu.npu_rms_norm`` and PyTorch 2.13 has ``F.rms_norm``; both fuse the chain into one kernel. This
script measures them on the real token count and checks the output against the reference, so a speed win is not
mistaken for a semantic change.

    python scripts/bench_norm_variants.py --tokens 113664 --dim 2048 --iters 50

Output: one row per variant with median latency, GiB/s of the minimal traffic, peak HBM and relL2 vs reference.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, default=113664, help="32 samples x 3552 generation tokens.")
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def rel_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    actual = actual.float()
    reference = reference.float()
    return float(((actual - reference).norm() / reference.norm()).item())


def timed(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        torch.npu.synchronize()
        started = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def main():
    import torch_npu  # noqa: F401
    from diffusers.models.transformers.transformer_cosmos3 import Cosmos3NemotronRMSNorm

    args = parse_args()
    torch.npu.set_device(0)
    dtype = getattr(torch, args.dtype)
    weight = torch.ones(args.dim, device="npu:0", dtype=dtype)
    x = torch.randn(args.tokens, args.dim, device="npu:0", dtype=dtype)

    reference = Cosmos3NemotronRMSNorm(args.dim, eps=args.eps).to("npu:0", dtype=dtype)
    with torch.no_grad():
        reference.weight.copy_(weight)
        expected = reference(x)
    torch.npu.synchronize()

    variants = {}
    variants["nemotron_reference"] = lambda: reference(x)

    if hasattr(torch.nn.functional, "rms_norm"):
        variants["torch_rms_norm"] = lambda: torch.nn.functional.rms_norm(x, (args.dim,), weight, args.eps)

    if hasattr(torch_npu, "npu_rms_norm"):
        variants["npu_rms_norm"] = lambda: torch_npu.npu_rms_norm(x, weight, args.eps)[0]

    if hasattr(torch_npu, "npu_add_rms_norm"):
        residual = torch.randn_like(x)

        def add_variant():
            return torch_npu.npu_add_rms_norm(residual, x, weight, args.eps)

        variants["npu_add_rms_norm"] = add_variant

    traffic = x.numel() * x.element_size() + weight.numel() * weight.element_size() \
        + x.numel() * x.element_size()
    print(f"tokens={args.tokens} dim={args.dim} dtype={args.dtype} minimal_traffic={traffic / 2**20:.1f} MiB")
    print(f"{'variant':>22} {'median ms':>10} {'GiB/s':>8} {'relL2':>10} {'peak GiB':>9}")

    for name, fn in variants.items():
        torch.npu.reset_peak_memory_stats()
        with torch.no_grad():
            out = fn()
            torch.npu.synchronize()
            candidate = out[0] if isinstance(out, (tuple, list)) else out
            error = rel_l2(candidate, expected)
            wall = timed(fn, args.warmup, args.iters)
        print(f"{name:>22} {wall * 1e3:10.3f} {traffic / wall / 2**30:8.1f} {error:10.2e} "
              f"{torch.npu.max_memory_allocated() / 2**30:9.2f}")

    import os

    os._exit(0)


if __name__ == "__main__":
    main()

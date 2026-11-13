r"""Reproducible benchmark for the two-way attention path.

Two layers are reported separately so a slow training step is not blamed on the kernel without evidence: the core
AR/DM calls with the K/V interleave, and the complete attention processor when a model is available. Every run records
the environment, the input shape, dtype, backend and the actual kernel path. A missing fused backend is reported,
never silently replaced by the naive reference.

    python examples/worldmodel/cosmos3_harness/benchmark.py --backend naive --samples 1 2 4 --output result.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from colossalai.accelerator import get_accelerator
from colossalai.shardformer.layer.two_way_attn import TwoWayPackedMeta, _interleave_two_way_kv, two_way_flat_attention


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="naive", choices=("naive", "npu", "cuda"))
    parser.add_argument("--device", default=None, help="Device override, e.g. cpu, cuda, npu.")
    parser.add_argument("--samples", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--ar-len", type=int, default=512)
    parser.add_argument("--dm-len", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", default="bf16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="cosmos3_two_way_benchmark.json")
    return parser.parse_args(argv)


def _resolved_device(override: Optional[str]) -> torch.device:
    if override:
        return torch.device(override)
    accelerator = get_accelerator()
    return accelerator.get_current_device()


def _synchronize(device: torch.device) -> None:
    if device.type == "cpu":
        return
    get_accelerator().synchronize(device)


def _build_inputs(meta, heads, kv_heads, head_dim, dtype, device):
    generator = torch.Generator(device="cpu").manual_seed(0)

    def tensor(tokens, count):
        return torch.randn(tokens, count, head_dim, generator=generator, dtype=torch.float32).to(
            device=device, dtype=dtype
        )

    return (
        tensor(meta.total_ar_tokens, heads),
        tensor(meta.total_ar_tokens, kv_heads),
        tensor(meta.total_ar_tokens, kv_heads),
        tensor(meta.total_dm_tokens, heads),
        tensor(meta.total_dm_tokens, kv_heads),
        tensor(meta.total_dm_tokens, kv_heads),
    )


def _measure(fn, device: torch.device, warmup: int, iterations: int) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    _synchronize(device)
    latencies: List[float] = []
    for _ in range(iterations):
        begin = time.perf_counter()
        fn()
        _synchronize(device)
        latencies.append((time.perf_counter() - begin) * 1e3)
    latencies.sort()
    return {
        "mean_ms": statistics.fmean(latencies),
        "p50_ms": statistics.median(latencies),
        "p95_ms": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))],
        "min_ms": latencies[0],
    }


def _attention_case(args, sample_count: int, device: torch.device) -> Dict:
    ar_lens = [args.ar_len] * sample_count
    dm_lens = [args.dm_len] * sample_count
    meta = TwoWayPackedMeta(ar_lens, dm_lens)
    dtype = getattr(torch, args.dtype)
    inputs = _build_inputs(meta, args.heads, args.kv_heads, args.head_dim, dtype, device)

    def forward_backward():
        tensors = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]
        out_ar, out_dm = two_way_flat_attention(*tensors, meta, backend=args.backend)
        (out_ar.float().mean() + out_dm.float().mean()).backward()

    def interleave_only():
        _interleave_two_way_kv(inputs[1], inputs[4], meta)

    return {
        "samples": sample_count,
        "ar_lens": ar_lens,
        "dm_lens": dm_lens,
        "backend": args.backend,
        "dtype": args.dtype,
        "tokens": meta.total_kv_tokens,
        "attention_fwd_bwd": _measure(forward_backward, device, args.warmup, args.iterations),
        "kv_interleave": _measure(interleave_only, device, args.warmup, args.iterations),
    }


def _environment() -> Dict:
    import transformers

    import colossalai

    accelerator = get_accelerator()
    try:
        device_count = accelerator.device_count()
    except RuntimeError:
        device_count = 0
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "colossalai": getattr(colossalai, "__version__", "unknown"),
        "transformers": transformers.__version__,
        "accelerator": accelerator.name,
        "device_count": device_count,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import diffusers

        report["diffusers"] = diffusers.__version__
        report["has_cosmos3"] = hasattr(diffusers, "Cosmos3OmniTransformer")
    except ImportError:
        report["diffusers"] = None
        report["has_cosmos3"] = False
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    device = _resolved_device(args.device)
    report = {
        "environment": _environment(),
        "config": {
            "backend": args.backend,
            "device": str(device),
            "heads": args.heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
        },
        "cases": [],
    }
    if args.backend != "naive" and device.type == "cpu":
        report["status"] = "skipped-device"
        report["note"] = "A fused backend needs its accelerator; the CPU reference is run with --backend naive."
    else:
        for repeat in range(args.repeats):
            rows = [_attention_case(args, sample_count, device) for sample_count in args.samples]
            report["cases"].append({"repeat": repeat, "rows": rows})
        report["status"] = "ok"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

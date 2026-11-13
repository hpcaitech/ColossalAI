r"""Operator-level benchmarks for the Cosmos 3 training path.

Every arm is measured under the same conditions: ``--warmup`` untimed calls, then ``--repeats`` runs of
``--iterations`` calls, reporting the median and minimum milliseconds per call. Each report carries the same
environment block (git commit, torch/torch_npu/CANN versions, device) and the exact geometry, so numbers from
different arms, builds or backends are comparable.

    # Local smoke test (CPU).
    python examples/worldmodel/cosmos3_harness/bench_ops.py --op all --device cpu --num-tensors 8 --numel 4096

    # On the box, with the shard geometry of the real ZeRO-2 step.
    python examples/worldmodel/cosmos3_harness/bench_ops.py --op all --device npu \
        --num-tensors 512 --numel 131072 --output bench-optimizer.json

The end-to-end step benchmark lives in ``train.py --bench-json``: it reuses the real training loop, so it cannot
drift from what is trained.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import torch


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--op", default="all", choices=("unscale-clip", "adamw", "all"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "npu"))
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--num-tensors", type=int, default=64, help="Shards (unscale) or parameters (AdamW).")
    parser.add_argument("--numel", type=int, default=16384, help="Elements per shard/parameter.")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="Write the JSON report here in addition to printing it.")
    args = parser.parse_args(argv)
    args.dtype = getattr(torch, args.dtype)
    return args


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _resolve_device(requested: str) -> torch.device:
    if requested == "npu":
        # Importing the package registers the device type with torch; the module object itself is not needed.
        importlib.import_module("torch_npu")
        return torch.device("npu")
    if requested != "auto":
        return torch.device(requested)
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "npu":
        return torch.npu.get_device_name(device)
    return platform.processor() or platform.machine()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu":
        torch.npu.synchronize()


def _cann_version() -> Optional[str]:
    """Read the CANN version the container actually ships, not the one the torch_npu wheel was built against."""
    home = os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest")
    try:
        text = (Path(home) / "version.cfg").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    match = re.search(r"^version\s*=\s*(\S+)", text, flags=re.MULTILINE)
    return match.group(1) if match else None


def collect_env(device: torch.device) -> Dict:
    """The block every report carries; ``train.py --bench-json`` reuses it so step and op numbers match."""
    env = {
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device_type": device.type,
        "device_name": _device_name(device),
    }
    if device.type == "npu":
        import torch_npu

        env["torch_npu"] = getattr(torch_npu, "__version__", "unknown")
        env["cann"] = _cann_version()
    return env


def measure(
    call: Callable[[], None],
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
) -> Dict:
    """Time ``call`` identically for every arm: warm up first, then take the median and minimum of the repeats."""
    for _ in range(warmup):
        call()
    runs = []
    for _ in range(repeats):
        _synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            call()
        _synchronize(device)
        runs.append((time.perf_counter() - start) * 1e3 / iterations)
    ordered = sorted(runs)
    return {
        "median_ms": ordered[len(ordered) // 2],
        "min_ms": ordered[0],
        "runs_ms": runs,
    }


class _UnscaleStub:
    """Only the attributes ``_unscale_and_clip_grads`` reads."""

    mixed_precision_mixin = None
    _clip_grad_norm = 0.0


def _bench_unscale_clip(args: argparse.Namespace, device: torch.device) -> Dict:
    from colossalai.zero.low_level.low_level_optim import LowLevelZeroOptimizer

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    shards = [
        torch.randn(args.numel, generator=generator).to(device=device, dtype=args.dtype)
        for _ in range(args.num_tensors)
    ]

    def per_tensor_loop():
        # The pre-change implementation: one launch per shard.
        for shard in shards:
            shard.data.mul_(1.0)

    def multi_tensor():
        LowLevelZeroOptimizer._unscale_and_clip_grads(_UnscaleStub(), shards, 0.0)

    timing = dict(warmup=args.warmup, iterations=args.iterations, repeats=args.repeats)
    return {
        "conditions": {"scale": 1.0, "note": "the scale factor does not change the kernel work"},
        "per_tensor_loop": measure(per_tensor_loop, device, **timing),
        "foreach_multi_tensor": measure(multi_tensor, device, **timing),
    }


def _bench_adamw(args: argparse.Namespace, device: torch.device) -> Dict:
    try:
        from examples.worldmodel.cosmos3_harness.optim import build_adamw
    except ImportError:  # direct execution from the example directory
        from optim import build_adamw

    backend = {"npu": "npu", "cuda": "cuda"}.get(device.type, "naive")
    timing = dict(warmup=args.warmup, iterations=args.iterations, repeats=args.repeats)
    arms = {}
    for mode in ("plain", "foreach", "fused", "hybrid"):
        if mode in ("fused", "hybrid") and device.type != "npu":
            # Both only exist as meaningful arms on the NPU box: the vendor fused kernel and the recipe's
            # per-parameter torch fallback, which is what the NPU branch of HybridAdam.step runs.
            continue
        torch.manual_seed(args.seed)
        params = [
            torch.nn.Parameter(torch.randn(args.numel, device=device, dtype=args.dtype))
            for _ in range(args.num_tensors)
        ]
        grads = [torch.randn_like(param) for param in params]
        if mode == "hybrid":
            # What the current SFT recipe uses: on NPU its step() falls into a per-parameter Python loop.
            from colossalai.nn.optimizer import HybridAdam

            optimizer = HybridAdam(
                params,
                lr=args.learning_rate,
                betas=tuple(args.betas),
                eps=args.eps,
                weight_decay=args.weight_decay,
                adamw_mode=True,
            )
        else:
            optimizer = build_adamw(
                params,
                args.learning_rate,
                backend,
                mode,
                betas=args.betas,
                eps=args.eps,
                weight_decay=args.weight_decay,
            )

        def step(params=params, grads=grads, optimizer=optimizer):
            for param, grad in zip(params, grads):
                param.grad = grad
            optimizer.step()

        arms[mode] = {
            "optimizer_class": type(optimizer).__name__,
            "foreach": optimizer.defaults.get("foreach"),
            **measure(step, device, **timing),
        }
    return arms


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    conditions = {
        "num_tensors": args.num_tensors,
        "numel": args.numel,
        "dtype": str(args.dtype).replace("torch.", ""),
        "learning_rate": args.learning_rate,
        "betas": list(args.betas),
        "eps": args.eps,
        "weight_decay": args.weight_decay,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "seed": args.seed,
    }

    results = {}
    if args.op in ("unscale-clip", "all"):
        results["unscale_clip"] = _bench_unscale_clip(args, device)
    if args.op in ("adamw", "all"):
        results["adamw"] = _bench_adamw(args, device)

    report = {"mode": "operator", "env": collect_env(device), "conditions": conditions, "results": results}
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

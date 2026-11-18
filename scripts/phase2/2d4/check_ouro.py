#!/usr/bin/env python3
"""2D4: shared Ouro weights under the current branch's ZeRO / Gemini plugins."""
import argparse
import hashlib
import json
import math
import os
import subprocess
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import bootstrap_numerics
import torch
import torch.distributed as dist
from ouro_numerics import entry

import colossalai
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin
from colossalai.testing import spawn
from tests.kit.numerics import Trace, judge
from tests.kit.numerics.judge import judge_relative_bound, require_gradients
from tests.kit.numerics.scenarios import data_parallel, independent_impl, resume


def build_plugin(name):
    if name.startswith("gemini"):
        # Force real chunk partitioning even for tiny models (four or more chunks).
        degree = dist.get_world_size()
        chunks = None if name.endswith("_default") else {degree: dict(chunk_size=32768, keep_gathered=False)}
        return GeminiPlugin(precision=name.split("_")[1], enable_gradient_accumulation=True, chunk_config_dict=chunks)
    stage = int(name[4])
    return LowLevelZeroPlugin(stage=stage, precision=name.split("_")[1], cpu_offload=name.endswith("_offload"))


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def save_report(report, path, **metadata):
    payload = dict(
        status="passed" if report.passed else "failed",
        compared_tensors=len(report.entries),
        failures=len(report.failures),
        entries=[asdict(item) for item in report.entries],
        **metadata,
    )
    Path(path).write_text(json.dumps(clean_json(payload), indent=2, ensure_ascii=False))
    print(report.format(limit=12), flush=True)
    return payload


def independent_worker(rank, world_size, port, args):
    torch.set_num_threads(2)
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)
    try:
        model_entry = entry(args.model, args.loops, args.size)
        if args.scenario == "blank":
            report = independent_impl.blank_control(model_entry, steps=args.steps, seed=args.seed)
        else:
            test, ref, truth = independent_impl.training_traces(
                model_entry, build_plugin(args.plugin), steps=args.steps, seed=args.seed
            )
            for role, trace in (("test", test), ("ref", ref), ("truth", truth)):
                trace.save(Path(args.out) / f"{role}-rank{rank}.safetensors")
            report = require_gradients(judge(test, ref, truth), test=test, ref=ref, truth=truth)
            if args.loops > 1:
                for name in ("model.early_exit_gate.weight", "model.early_exit_gate.bias"):
                    gradient = test[f"grad/param/{name}"]
                    if not torch.isfinite(gradient).all() or gradient.norm().item() == 0:
                        raise AssertionError(f"Loop gate did not receive a finite, nonzero gradient: {name}")
        save_report(report, Path(args.out) / f"rank{rank}.json", rank=rank, world_size=world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def source_metadata():
    root = bootstrap_numerics.ROOT
    tracked = [
        "colossalai/shardformer/modeling/ouro.py",
        "tests/kit/remote_models.py",
        "colossalai/booster/plugin/low_level_zero_plugin.py",
        "colossalai/booster/plugin/gemini_plugin.py",
        "colossalai/zero/low_level/low_level_optim.py",
        "colossalai/zero/gemini/gemini_ddp.py",
        "colossalai/zero/gemini/gemini_optimizer.py",
    ]
    return dict(
        branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip(),
        server_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        numerics_commit=bootstrap_numerics.TOOL_COMMIT,
        numerics_root=str(bootstrap_numerics.TOOL_ROOT),
        source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in tracked},
        script_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
        torch=torch.__version__,
        cuda=torch.version.cuda,
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        note="Model, plugins and numerical tools all use the same mounted working tree.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["ouro14", "ouro26"], default="ouro14")
    parser.add_argument("--loops", type=int, choices=[1, 2, 4], default=4)
    parser.add_argument("--size", choices=["tiny", "small"], default="tiny")
    parser.add_argument(
        "--scenario", choices=["blank", "independent", "dp", "dp_tokens", "resume"], default="independent"
    )
    parser.add_argument(
        "--plugin",
        choices=[
            "zero1_fp32",
            "zero1_bf16",
            "zero2_fp32",
            "zero2_bf16",
            "zero2_bf16_offload",
            "gemini_bf16",
            "gemini_bf16_default",
        ],
        default="zero2_fp32",
    )
    parser.add_argument("--world-size", type=int, choices=[1, 2], default=1)
    parser.add_argument("--micro-batches", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.scenario == "blank":
        # The shared tool fixes this control to one-rank ZeRO-1 BF16 on CUDA.
        args.plugin = "zero1_bf16"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    metadata = dict(
        created=datetime.now(timezone.utc).isoformat(),
        case=vars(args),
        source=source_metadata(),
        scope="tiny random initialization; no official pretrained weights or full-size claim",
    )
    (Path(args.out) / "case.json").write_text(json.dumps(metadata, indent=2))
    print("CASE", json.dumps(vars(args)), flush=True)
    try:
        if args.scenario in ("independent", "blank"):
            if args.scenario == "blank" and args.world_size != 1:
                raise ValueError("blank control requires one rank")
            spawn(independent_worker, args.world_size, args=args)
            ranks = [json.loads((Path(args.out) / f"rank{i}.json").read_text()) for i in range(args.world_size)]
            result = dict(status="passed" if all(x["status"] == "passed" for x in ranks) else "failed", ranks=ranks)
        elif args.scenario in ("dp", "dp_tokens"):
            test, ref = data_parallel.traces(
                entry(args.model, args.loops, args.size),
                partial(build_plugin, args.plugin),
                world_size=args.world_size,
                micro_batches=args.micro_batches,
                seed=args.seed,
                directory=str(Path(args.out) / "traces"),
                token_weighted=args.scenario == "dp_tokens",
            )
            subset = lambda trace: Trace({k: v for k, v in trace.items() if k.startswith("grad/")}, trace.meta)
            report = require_gradients(
                judge_relative_bound(subset(test), subset(ref), torch.bfloat16), test=test, ref=ref
            )
            result = save_report(report, Path(args.out) / "alignment.json")
        else:
            report = resume.check(
                entry(args.model, args.loops, args.size),
                partial(build_plugin, args.plugin),
                world_size=args.world_size,
                seed=args.seed,
                directory=str(Path(args.out) / "traces"),
            )
            result = save_report(report, Path(args.out) / "alignment.json")
        result.update(metadata)
    except Exception as error:
        traceback.print_exc()
        result = dict(
            status="error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(), **metadata
        )
    (Path(args.out) / "report.json").write_text(json.dumps(clean_json(result), indent=2, ensure_ascii=False))
    print("RESULT", result["status"], flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

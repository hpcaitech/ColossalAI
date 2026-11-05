"""L2 acceptance report of a model_zoo entry: every check under every configuration, written to one JSON file.

The judging rules are the tool's defaults and cannot be changed from here. Each result is ``passed``, ``failed``
(including a run that raised), ``unsupported`` (the tool refuses the configuration, e.g. an unverified precision
recipe) or ``not_run`` (this machine cannot run it). The report records the repository commit (and whether the tree
was dirty), the command and the environment, so it can be rerun and audited.

    python -m tests.kit.numerics.acceptance --entry transformers_mixtral --out mixtral.json
    python -m tests.kit.numerics.acceptance --entry transformers_mixtral --configs ddp_fp32,zero2_bf16 \\
        --checks independent_impl,resume
    python -m tests.kit.numerics.acceptance --entry <real-weight entry> --cache-dir <dir>  # reuse the float64 truth

Depends on ColossalAI, so the package ``__init__`` does not import it.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
from torch.distributed.fsdp import MixedPrecision

import colossalai
from colossalai.booster.mixed_precision import MixedPrecision as BoosterMixedPrecision
from colossalai.booster.mixed_precision.fp16_torch import FP16TorchMixedPrecision
from colossalai.accelerator import get_accelerator
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.booster.plugin.torch_fsdp_plugin import TorchFSDPPlugin
from colossalai.testing import rerun_if_address_is_in_use, spawn

from .capture.recipes import UnsupportedRecipe
from .judge import AlignmentReport
from .scenarios import data_parallel, independent_impl, resume
from .spec import NumericsSpec
from .trace import environment_meta

SCHEMA_VERSION = 1
PASSED, FAILED, UNSUPPORTED, NOT_RUN = "passed", "failed", "unsupported", "not_run"


@dataclass(frozen=True)
class Config:
    """A plugin and precision to check under. ``plugin_fn`` builds the plugin inside each process."""

    plugin_fn: Callable
    mixed_precision: Union[str, BoosterMixedPrecision, None] = None
    # needs a CUDA or NPU device; the data-parallel check needs two
    needs_device: bool = True
    # resume: loading a sharded optimizer fails on CPU (see scenarios.resume)
    shard_optimizer: bool = True


FSDP_BF16 = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
CONFIGS: Dict[str, Config] = {
    "ddp_fp32": Config(TorchDDPPlugin, needs_device=False),
    "ddp_fp16": Config(TorchDDPPlugin, "fp16"),
    # BF16Module uses the accelerator's autocast; CPUAccelerator does not implement BF16 autocast.
    "ddp_bf16": Config(TorchDDPPlugin, "bf16"),
    "zero1_fp32": Config(partial(LowLevelZeroPlugin, stage=1, precision="fp32")),
    "zero1_bf16": Config(partial(LowLevelZeroPlugin, stage=1, precision="bf16")),
    "zero2_fp32": Config(partial(LowLevelZeroPlugin, stage=2, precision="fp32")),
    "zero2_bf16": Config(partial(LowLevelZeroPlugin, stage=2, precision="bf16")),
    "zero2_bf16_offload": Config(partial(LowLevelZeroPlugin, stage=2, precision="bf16", cpu_offload=True)),
    # shardformer policy with tp_size=1: needs a model the policies support
    "shardformer_fp32": Config(partial(HybridParallelPlugin, tp_size=1, pp_size=1, precision="fp32")),
    "shardformer_bf16": Config(partial(HybridParallelPlugin, tp_size=1, pp_size=1, precision="bf16")),
    # gradient accumulation on: the single-device side of data_parallel accumulates micro-batches
    "gemini_bf16": Config(partial(GeminiPlugin, precision="bf16", enable_gradient_accumulation=True)),
    "gemini_fp16": Config(partial(GeminiPlugin, precision="fp16", enable_gradient_accumulation=True)),
    "fsdp_fp32": Config(TorchFSDPPlugin),
    "fsdp_bf16": Config(partial(TorchFSDPPlugin, mixed_precision=FSDP_BF16)),
}
CHECKS = ("blank_control", "independent_impl", "data_parallel", "data_parallel_tokens", "resume")
# which L2 item of the roadmap each check answers
L2_ITEMS = {
    "blank_control": "precondition of ① ⑤",
    "independent_impl": "① ⑤",
    "data_parallel": "②",
    "data_parallel_tokens": "② with the model's loss normalized by valid tokens",
    "resume": "④",
}


def _entry_config(name: str, spec: NumericsSpec) -> Config:
    config = CONFIGS[name]
    scale = spec.fp16_loss_scale
    if scale is None:
        return config
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("NumericsSpec.fp16_loss_scale must be positive and finite")
    if name == "ddp_fp16":
        return replace(config, mixed_precision=FP16TorchMixedPrecision(init_scale=scale))
    if name == "gemini_fp16":
        return replace(config, plugin_fn=partial(config.plugin_fn, initial_scale=scale))
    return config


def run_acceptance(
    entry: Union[str, tuple],
    configs: Sequence[str] = tuple(CONFIGS),
    checks: Sequence[str] = CHECKS,
    steps: int = 3,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run ``checks`` under ``configs`` for ``entry`` (a model_zoo name, or a picklable entry tuple) and return the
    report as a JSON-serializable dict. Call from the main process: every check spawns its own process group.

    ``data_parallel_tokens`` runs only for entries that declare ``NumericsSpec.token_loss`` (``not_run`` otherwise).
    ``cache_dir`` keeps the reference runs of ``independent_impl`` and ``blank_control`` (see ``capture.cache``)."""
    unknown = sorted(set(configs) - set(CONFIGS)) + sorted(set(checks) - set(CHECKS))
    if unknown:
        raise ValueError(f"unknown configs or checks: {unknown}; known: {sorted(CONFIGS)}, {list(CHECKS)}")
    results = []
    spec = _spec(entry)
    token_loss = spec.token_loss is not None
    if "blank_control" in checks:
        run = partial(_in_process_group, _blank_control, entry, steps, cache_dir)
        results.append(_result("blank_control", None, run))
    for name in configs:
        config = _entry_config(name, spec)
        for check in checks:
            if check == "blank_control":
                continue
            missing = _missing_devices(config, 2 if check.startswith("data_parallel") else 1)
            if check == "data_parallel_tokens" and not token_loss:
                missing = "the entry declares no NumericsSpec.token_loss"
            if missing:
                results.append(_entry_result(check, name, NOT_RUN, reason=missing))
            elif check == "independent_impl":
                run = partial(_in_process_group, _independent_impl, entry, steps, config, cache_dir)
                results.append(_result(check, name, run))
            elif check.startswith("data_parallel"):
                run = partial(
                    data_parallel.check,
                    entry,
                    config.plugin_fn,
                    config.mixed_precision,
                    world_size=2,
                    micro_batches=4,
                    token_weighted=check == "data_parallel_tokens",
                )
                results.append(_result(check, name, run))
            else:
                shard = config.shard_optimizer and get_accelerator().name != "cpu"
                if not shard and not resume.unsharded_optimizer_reloads():
                    reason = "ColossalAI cannot load back its unsharded optimizer checkpoints with this torch"
                    results.append(_entry_result(check, name, NOT_RUN, reason=reason))
                    continue
                run = partial(resume.check, entry, config.plugin_fn, config.mixed_precision, shard_optimizer=shard)
                results.append(_result(check, name, run))
    return {
        "schema_version": SCHEMA_VERSION,
        "entry": entry if isinstance(entry, str) else "custom",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": sys.argv,
        "repository": _repository(),
        "environment": _environment(),
        "steps": steps,
        "recipe_overrides": {"fp16_initial_loss_scale": spec.fp16_loss_scale},
        "results": results,
        "summary": {s: sum(r["status"] == s for r in results) for s in (PASSED, FAILED, UNSUPPORTED, NOT_RUN)},
    }


def format_report(report: Dict[str, Any]) -> str:
    """One line per result, for reading the JSON at a glance."""
    lines = [f"{report['entry']} @ {report['repository'].get('commit', '?')[:8]}"]
    if report["repository"].get("dirty"):
        lines[0] += " (dirty tree)"
    lines.append(f"{'check':<18} {'config':<20} {'status':<12} detail")
    for r in report["results"]:
        detail = r.get("reason", "")
        if r["status"] in (PASSED, FAILED) and "tensors" in r:
            detail = f"{r['tensors'] - len(r['failures'])}/{r['tensors']} tensors, max ratio {r['max_ratio']}"
            if r.get("routing"):
                detail += f", routing flips test {r['routing']['test']} / ref {r['routing']['ref']}"
            if r["failures"]:
                detail += "; " + ", ".join(f"{f['name']} ({f['reason']})" for f in r["failures"][:3])
        lines.append(f"{r['check']:<18} {r['config'] or '-':<20} {r['status']:<12} {detail}")
    lines.append(", ".join(f"{n} {s}" for s, n in report["summary"].items()))
    return "\n".join(lines)


def _result(check: str, config: Optional[str], run: Callable[[], AlignmentReport]) -> Dict[str, Any]:
    try:
        # a port taken between picking and binding it is not a result; pytest runs get the same retry
        report = rerun_if_address_is_in_use()(run)()
    except Exception as exc:  # a spawned process re-raises as ProcessRaisedException with the original text
        text = f"{type(exc).__name__}: {exc}"
        if "UnsupportedRecipe" in text or isinstance(exc, UnsupportedRecipe):
            return _entry_result(check, config, UNSUPPORTED, reason=_last_line(text))
        return _entry_result(check, config, FAILED, reason=_last_line(text), error=traceback.format_exc()[-4000:])
    failures = [_describe(e) for e in report.failures]
    ratios = [e.ratio for e in report.entries if e.passed and _finite(e.ratio)]
    return _entry_result(
        check,
        config,
        PASSED if report.passed else FAILED,
        rule=report.baseline_label,
        tensors=len(report.entries),
        max_ratio=round(max(ratios), 4) if ratios else None,
        failures=failures,
        **({"routing": report.routing} if report.routing is not None else {}),
    )


def _entry_result(check: str, config: Optional[str], status: str, **fields) -> Dict[str, Any]:
    return {"check": check, "l2_item": L2_ITEMS[check], "config": config, "status": status, **fields}


def _describe(entry) -> Dict[str, Any]:
    return {
        "name": entry.name,
        "reason": entry.reason,
        "allowed": _number(entry.allowed_error),
        "error": _number(entry.test_error),
        "ratio": _number(entry.ratio),
    }


def _number(x: float) -> Optional[float]:
    return float(f"{x:.4g}") if _finite(x) else None


def _finite(x: float) -> bool:
    return isinstance(x, float) and math.isfinite(x)


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else text


def _missing_devices(config: Config, needed: int) -> Optional[str]:
    accelerator = get_accelerator()
    if accelerator.name == "cpu":
        if config.needs_device:
            return "needs a CUDA or NPU device"
        return None
    if accelerator.device_count() < needed:
        return f"needs {needed} devices, found {accelerator.device_count()}"
    return None


def _in_process_group(fn: Callable, *args) -> AlignmentReport:
    # independent_impl runs inside one process with torch.distributed initialized; the report comes back as a file
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "report.pt")
        spawn(_worker, 1, fn=fn, args=args, path=path)
        return torch.load(path, weights_only=False)


def _worker(rank, world_size, port, fn, args, path):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)
    torch.save(fn(*args), path)


def _blank_control(entry, steps, cache_dir) -> AlignmentReport:
    return independent_impl.blank_control(entry, steps=steps, cache_dir=cache_dir)


def _independent_impl(entry, steps, config: Config, cache_dir) -> AlignmentReport:
    plugin = config.plugin_fn()
    return independent_impl.check_training(entry, plugin, config.mixed_precision, steps=steps, cache_dir=cache_dir)


def _spec(entry) -> NumericsSpec:
    if isinstance(entry, str):
        from tests.kit.model_zoo import model_zoo

        entry = model_zoo[entry]
    attribute = entry[4]
    return (attribute.numerics if attribute is not None else None) or NumericsSpec()


def _repository() -> Dict[str, Any]:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    def git(*args) -> str:
        # safe.directory: the checkout may belong to another user than the one running (e.g. root in a container)
        command = ["git", "-c", "safe.directory=*", *args]
        return subprocess.run(command, cwd=root, capture_output=True, text=True, check=True).stdout.strip()

    try:
        changed = git("status", "--porcelain").splitlines()
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(changed), "changed": changed[:50]}
    except subprocess.CalledProcessError as exc:
        # e.g. a worktree mounted into a container without the main checkout its .git file points to
        return {"error": f"{exc}: {exc.stderr.strip()}"}
    except OSError as exc:
        return {"error": str(exc)}


def _environment() -> Dict[str, Any]:
    accelerator = get_accelerator()
    meta = environment_meta()
    meta.update(
        colossalai=getattr(colossalai, "__version__", "unknown"),
        accelerator=accelerator.name,
        device_count=accelerator.device_count() if accelerator.name != "cpu" else 0,
    )
    if accelerator.name == "cuda":
        meta["device_name"] = torch.cuda.get_device_name(0)
    return meta


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--entry", required=True, help="model_zoo entry name, e.g. transformers_mixtral")
    parser.add_argument("--configs", default=",".join(CONFIGS), help=f"comma-separated, from {', '.join(CONFIGS)}")
    parser.add_argument("--checks", default=",".join(CHECKS), help=f"comma-separated, from {', '.join(CHECKS)}")
    parser.add_argument("--steps", type=int, default=3, help="training steps of independent_impl (losses of each)")
    parser.add_argument("--out", help="JSON report path (default: numerics-<entry>.json)")
    parser.add_argument("--cache-dir", help="keep and reuse reference runs there (for real weights; see capture.cache)")
    args = parser.parse_args(argv)
    report = run_acceptance(
        args.entry, args.configs.split(","), args.checks.split(","), args.steps, cache_dir=args.cache_dir
    )
    out = args.out or f"numerics-{args.entry}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(format_report(report))
    print(f"report written to {out}")
    return 1 if report["summary"][FAILED] else 0


if __name__ == "__main__":
    sys.exit(main())

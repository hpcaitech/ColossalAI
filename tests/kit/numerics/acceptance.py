"""L2 acceptance report of a model_zoo entry: every check under every configuration, written to one JSON file.

Model lines run this themselves and hand in the JSON; the judging rules are the tool's defaults and cannot be changed
from here. Each result is ``passed``, ``failed`` (including a run that raised), ``unsupported`` (the tool refuses the
configuration, e.g. an unverified precision recipe) or ``not_run`` (this machine cannot run it). The report records the
repository commit (and whether the tree was dirty), the command and the environment, so it can be rerun and audited.

    python -m tests.kit.numerics.acceptance --entry transformers_mixtral --out mixtral.json
    python -m tests.kit.numerics.acceptance --entry transformers_mixtral --configs ddp_fp32,zero2_bf16 \\
        --checks independent_impl,resume

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
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster.plugin import HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.testing import rerun_if_address_is_in_use, spawn

from .capture.recipes import UnsupportedRecipe
from .judge import AlignmentReport
from .scenarios import data_parallel, independent_impl, resume
from .trace import environment_meta

SCHEMA_VERSION = 1
PASSED, FAILED, UNSUPPORTED, NOT_RUN = "passed", "failed", "unsupported", "not_run"


@dataclass(frozen=True)
class Config:
    """A plugin and precision to check under. ``plugin_fn`` builds the plugin inside each process."""

    plugin_fn: Callable
    mixed_precision: Optional[str] = None
    # needs a CUDA or NPU device; the data-parallel check needs two
    needs_device: bool = True
    # resume: loading a sharded optimizer fails on CPU (see scenarios.resume)
    shard_optimizer: bool = True


CONFIGS: Dict[str, Config] = {
    "ddp_fp32": Config(TorchDDPPlugin, needs_device=False),
    "ddp_fp16": Config(TorchDDPPlugin, "fp16"),
    # refused before anything runs: BF16MixedPrecision is not implemented
    "ddp_bf16": Config(TorchDDPPlugin, "bf16", needs_device=False),
    "zero1_fp32": Config(partial(LowLevelZeroPlugin, stage=1, precision="fp32")),
    "zero1_bf16": Config(partial(LowLevelZeroPlugin, stage=1, precision="bf16")),
    "zero2_fp32": Config(partial(LowLevelZeroPlugin, stage=2, precision="fp32")),
    "zero2_bf16": Config(partial(LowLevelZeroPlugin, stage=2, precision="bf16")),
    "zero2_bf16_offload": Config(partial(LowLevelZeroPlugin, stage=2, precision="bf16", cpu_offload=True)),
    # shardformer policy with tp_size=1: needs a model the policies support
    "shardformer_fp32": Config(partial(HybridParallelPlugin, tp_size=1, pp_size=1, precision="fp32")),
    "shardformer_bf16": Config(partial(HybridParallelPlugin, tp_size=1, pp_size=1, precision="bf16")),
}
CHECKS = ("blank_control", "independent_impl", "data_parallel", "resume")
# which L2 item of the roadmap each check answers
L2_ITEMS = {
    "blank_control": "precondition of ① ⑤",
    "independent_impl": "① ⑤",
    "data_parallel": "②",
    "resume": "④",
}


def run_acceptance(
    entry: Union[str, tuple],
    configs: Sequence[str] = tuple(CONFIGS),
    checks: Sequence[str] = CHECKS,
    steps: int = 3,
) -> Dict[str, Any]:
    """Run ``checks`` under ``configs`` for ``entry`` (a model_zoo name, or a picklable entry tuple) and return the
    report as a JSON-serializable dict. Call from the main process: every check spawns its own process group."""
    unknown = sorted(set(configs) - set(CONFIGS)) + sorted(set(checks) - set(CHECKS))
    if unknown:
        raise ValueError(f"unknown configs or checks: {unknown}; known: {sorted(CONFIGS)}, {list(CHECKS)}")
    results = []
    if "blank_control" in checks:
        results.append(_result("blank_control", None, lambda: _in_process_group(_blank_control, entry, steps)))
    for name in configs:
        config = CONFIGS[name]
        for check in checks:
            if check == "blank_control":
                continue
            missing = _missing_devices(config, 2 if check == "data_parallel" else 1)
            if missing:
                results.append(_entry_result(check, name, NOT_RUN, reason=missing))
            elif check == "independent_impl":
                run = partial(_in_process_group, _independent_impl, entry, steps, config)
                results.append(_result(check, name, run))
            elif check == "data_parallel":
                run = partial(data_parallel.check, entry, config.plugin_fn, config.mixed_precision, world_size=2)
                results.append(_result(check, name, run))
            else:
                shard = config.shard_optimizer and get_accelerator().name != "cpu"
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


def _blank_control(entry, steps) -> AlignmentReport:
    return independent_impl.blank_control(entry, steps=steps)


def _independent_impl(entry, steps, config: Config) -> AlignmentReport:
    return independent_impl.check_training(entry, config.plugin_fn(), config.mixed_precision, steps=steps)


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
    args = parser.parse_args(argv)
    report = run_acceptance(args.entry, args.configs.split(","), args.checks.split(","), args.steps)
    out = args.out or f"numerics-{args.entry}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(format_report(report))
    print(f"report written to {out}")
    return 1 if report["summary"][FAILED] else 0


if __name__ == "__main__":
    sys.exit(main())

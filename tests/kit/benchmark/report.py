"""Benchmark report: the JSON every run writes, its summary statistics, and the one Markdown template all lines use.

    python -m tests.kit.benchmark.report result.json > result.md

The JSON keeps every timed step of every repeat, so any statistic can be recomputed later; the Markdown is rendered
from it and nothing else.
"""

import argparse
import json
import math
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence

from .flops import FLOPS_MISMATCH_LIMIT

SCHEMA_VERSION = 2
# two repeats of the same setup whose median step times differ by more than this are flagged: the machine was not
# stable enough for the numbers to be compared with anything
REPEAT_SPREAD_LIMIT = 0.05
GiB = 1024**3


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile (0-100), linearly interpolated between the closest ranks (numpy's default)."""
    if not values:
        raise ValueError("percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(run: Dict[str, Any], flops: Dict[str, Any], peak: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Step time statistics, throughput and utilization of one repeat, from its raw step times.

    ``flops`` is ``flops.flops_per_token(...)``; ``peak`` a ``peak.Peak`` as a dict, whose TFLOPS are None when the
    device's peak is not known (no MFU then)."""
    times = run["step_time_s"]
    median = statistics.median(times)
    per_device = run["tokens_per_step"] / median / run["world_size"]
    summary = {
        "step_time_s": {"median": median, "p10": percentile(times, 10), "p90": percentile(times, 90)},
        # from the median step: one slow step (a stall, a collective that waited) does not move the headline
        "tokens_per_s": run["tokens_per_step"] / median,
        "tokens_per_s_per_device": per_device,
        "samples_per_s": run["samples_per_step"] / median,
        "model_tflops_per_device": flops["model_flops_per_token"] * per_device / 1e12,
        "hardware_tflops_per_device": flops["hardware_flops_per_token"] * per_device / 1e12,
    }
    for name, tflops in (("mfu", summary["model_tflops_per_device"]), ("hfu", summary["hardware_tflops_per_device"])):
        for kind in ("nominal", "measured"):
            denominator = peak[f"{kind}_tflops"]
            summary[f"{name}_{kind}"] = tflops / denominator if denominator else None
    return summary


def build_report(
    runs: List[Dict[str, Any]],
    environment: Dict[str, Any],
    repository: Dict[str, Any],
    command: List[str],
    flops: Dict[str, Any],
    peak: Dict[str, Any],
) -> Dict[str, Any]:
    """The report of one setup measured ``len(runs)`` times, each in fresh processes (``measure.measure`` results),
    with the FLOPs per token (``flops.flops_per_token``) and the device's peak (see ``summarize``)."""
    if not runs:
        raise ValueError("a report needs at least one run")
    setups = {json.dumps(r["setup"], sort_keys=True) for r in runs}
    if len(setups) != 1:
        raise ValueError("every run of a report must measure the same setup")
    summaries = [summarize(r, flops, peak) for r in runs]
    medians = [s["step_time_s"]["median"] for s in summaries]
    spread = (max(medians) - min(medians)) / min(medians)
    # the repeat with the median throughput; with an even count the slower of the two middle ones
    order = sorted(range(len(runs)), key=lambda i: medians[i])
    headline = order[len(order) // 2]
    flags = []
    if spread > REPEAT_SPREAD_LIMIT:
        flags.append(
            f"repeats differ by {spread:.1%} in median step time (limit {REPEAT_SPREAD_LIMIT:.0%}): the machine was "
            "not stable, do not compare these numbers"
        )
    if not all(r["loss"]["finite"] for r in runs):
        flags.append("a loss was not finite: the steps did not train, the timing may not be representative")
    mfu = summaries[headline]["mfu_nominal"]
    if mfu is not None and mfu > 1:
        flags.append(
            f"MFU is {mfu:.0%}, above the device's peak: the FLOPs per token overstate the work done (e.g. attention "
            "that the kernel skips), MFU is not reliable"
        )
    if flops["difference"] is not None and abs(flops["difference"]) > FLOPS_MISMATCH_LIMIT:
        flags.append(
            f"the FLOPs formula and the counted FLOPs differ by {flops['difference']:+.1%} (limit "
            f"{FLOPS_MISMATCH_LIMIT:.0%}): one of them does not describe the model, MFU is not reliable"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "setup": runs[0]["setup"],
        "world_size": runs[0]["world_size"],
        "parameters": runs[0]["parameters"],
        "optimizer": runs[0]["optimizer"],
        "headline_repeat": headline,
        "repeat_spread": spread,
        "flags": flags,
        "throughput": summaries[headline],
        "memory_bytes": {key: max(r["memory_bytes"].get(key, 0) for r in runs) for key in runs[0]["memory_bytes"]},
        "flops": flops,
        "peak": peak,
        "repeats": [dict(run, summary=summary) for run, summary in zip(runs, summaries)],
        "environment": environment,
        "repository": repository,
        "command": command,
    }


# metrics a report does not cover yet, listed in every report so that a missing number is never read as zero
NOT_COVERED = (
    "maximum sequence length on one device",
    "maximum micro-batch on one device",
    "data-parallel scaling efficiency",
    "comparison with a baseline",
)


def render_markdown(report: Dict[str, Any]) -> str:
    setup, throughput, memory = report["setup"], report["throughput"], report["memory_bytes"]
    env, repo = report["environment"], report["repository"]
    commit = repo.get("commit", "unknown")[:8] + (" (dirty tree)" if repo.get("dirty") else "")
    devices = f"{report['world_size']} × {env.get('device_name', env.get('accelerator', '?'))}"
    lines = [
        f"# Benchmark: {setup['entry']} ({setup['size']}), {setup['config']}",
        "",
        "## Summary",
        "",
        f"{_tokens(throughput['tokens_per_s'])} tokens/s on {devices} "
        f"({_tokens(throughput['tokens_per_s_per_device'])} per device), "
        f"MFU {_utilization(throughput, 'mfu', report['peak'])}, "
        f"peak reserved memory {_gib(memory.get('reserved_peak'))}. Commit {commit}.",
        "",
    ]
    if "search" in report:
        search = report["search"]
        lines[-1:-1] = [f"Maximum {SEARCH_NAMES[search['field']]} on one device: **{search['max']}**."]
    lines += [f"> **Warning:** {flag}" for flag in report["flags"]] + ([""] if report["flags"] else [])
    lines += [
        "## Configuration",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Entry / size | {setup['entry']} / {setup['size']} ({report['parameters'] / 1e9:.2f} B parameters) |",
        f"| Plugin and precision | {setup['config']} |",
        f"| Optimizer | {report['optimizer']} |",
        f"| Devices | {devices} |",
        f"| Sequence length × micro-batch per device | {setup['seq_len']} × {setup['micro_batch']} |",
        f"| Gradient checkpointing | {'on (non-reentrant)' if setup['gradient_checkpointing'] else 'off'} |",
        f"| Steps | {setup['warmup_steps']} warmup + {setup['timed_steps']} timed, "
        f"{len(report['repeats'])} repeats in fresh processes |",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Definition |",
        "| --- | --- | --- |",
        f"| Throughput | {_tokens(throughput['tokens_per_s'])} tokens/s | all devices, from the median step time |",
        f"| Throughput per device | {_tokens(throughput['tokens_per_s_per_device'])} tokens/s | |",
        f"| Samples | {throughput['samples_per_s']:.2f} /s | |",
        f"| MFU | {_utilization(throughput, 'mfu', report['peak'])} | model FLOPs per token × tokens/s per device ÷ "
        "peak; no recomputation counted |",
        f"| Model TFLOPS per device | {throughput['model_tflops_per_device']:.1f} | |",
        f"| Step time | {throughput['step_time_s']['median'] * 1e3:.1f} ms "
        f"(p10 {throughput['step_time_s']['p10'] * 1e3:.1f}, p90 {throughput['step_time_s']['p90'] * 1e3:.1f}) | "
        "slowest rank per step, device synchronized before and after |",
        f"| Repeat spread | {report['repeat_spread']:.1%} | difference of the repeats' median step times |",
        f"| Peak allocated | {_gib(memory.get('allocated_peak'))} | tensors (`max_memory_allocated`) |",
        f"| Peak reserved | {_gib(memory.get('reserved_peak'))} | caching allocator (`max_memory_reserved`), "
        "decides out-of-memory |",
        f"| Peak on device | {_gib(memory.get('device_peak'))} of {_gib(memory.get('device_total'))} | reserved peak "
        "plus what the process holds outside the allocator (context, communication buffers) |",
        f"| After setup | {_gib(memory.get('setup_allocated'))} | allocated once model and optimizer are boosted |",
        f"| Host RSS peak | {_gib(memory.get('host_rss_peak'))} | per process, including model construction |",
        "",
        "Memory is the maximum over ranks.",
        "",
    ]
    if setup["gradient_checkpointing"]:
        after = next(i for i, line in enumerate(lines) if line.startswith("| Model TFLOPS per device")) + 1
        lines[after:after] = [
            f"| HFU | {_utilization(throughput, 'hfu', report['peak'])} | as MFU, with the forward that gradient "
            "checkpointing recomputes |"
        ]
    flops, peak = report["flops"], report["peak"]
    difference = "no formula" if flops["difference"] is None else f"{flops['difference']:+.2%}"
    lines += [
        "## FLOPs and peak",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Model FLOPs per token | {_gflop(flops['model_flops_per_token'])} (from the {flops['source']}) |",
        f"| Formula | {_gflop(flops['analytic'])} |",
        f"| Counted (`FlopCounterMode`, meta device) | {_gflop(flops['counted'])} ({difference} from the formula) |",
        f"| Hardware FLOPs per token | {_gflop(flops['hardware_flops_per_token'])} |",
        f"| Peak, nominal ({peak['device']}, {peak['dtype']}) | {_tflops(peak['nominal_tflops'])}: "
        f"{peak['nominal_source'] or 'unknown'} |",
        f"| Peak, measured | {_tflops(peak['measured_tflops'])}: {peak['measured_source'] or 'not measured'} |",
        "",
        *_search_section(report.get("search")),
        "## Data-parallel scaling",
        "",
        "Not measured by this report.",
        "",
        "## Baseline",
        "",
        "Not compared by this report.",
        "",
        "## Environment and reproduction",
        "",
        "| Item | Value |",
        "| --- | --- |",
    ]
    lines += [f"| {key} | {value} |" for key, value in env.items()]
    lines += [f"| commit | {commit} |", "", "```bash", " ".join(report["command"]), "```", "", "## Not covered", ""]
    missing = list(NOT_COVERED)
    if "search" in report:
        missing.remove(SEARCH_ITEMS[report["search"]["field"]])
    if peak["nominal_tflops"] is None:
        missing.insert(0, f"MFU: no peak is known for {peak['device']} in {peak['dtype']}")
    lines += [f"- {item}" for item in missing]
    return "\n".join(lines) + "\n"


def _tokens(value: float) -> str:
    return f"{value:,.0f}"


SEARCH_ITEMS = {"seq_len": NOT_COVERED[0], "micro_batch": NOT_COVERED[1]}
SEARCH_NAMES = {"seq_len": "sequence length", "micro_batch": "micro-batch"}


def _search_section(search: Optional[Dict[str, Any]]) -> List[str]:
    """The maximum found by ``search.run_search`` and every probe, in the order run."""
    if search is None:
        return []
    name = SEARCH_NAMES[search["field"]]
    fixed = ", ".join(f"{SEARCH_NAMES[k]} {v}" for k, v in search["fixed"].items())
    bound = f", at most {search['upper']}" if search["upper"] else ""
    lines = [
        f"## Maximum {name} on one device",
        "",
        f"**{search['max']}** (multiples of {search['granularity']}{bound}; {fixed}). Each probe trained "
        f"{search['probe_steps']} full steps, as many as the measurement, in a fresh process; the metrics above are "
        f"measured at {search['measured_at']}.",
        "",
        f"| {name.capitalize()} | Result | Peak reserved |",
        "| --- | --- | --- |",
    ]
    for probe in search["probes"]:
        result = "fits" if probe["fits"] else f"does not fit: {probe['reason']}"
        lines.append(f"| {probe['value']} | {result} | {_gib(probe.get('reserved_peak'))} |")
    return lines + [""]


def _utilization(throughput: Dict[str, Any], name: str, peak: Dict[str, Any]) -> str:
    nominal, measured = throughput[f"{name}_nominal"], throughput[f"{name}_measured"]
    if nominal is None:
        return "n/a (no peak known)"
    text = f"{nominal:.1%} of the nominal {peak['nominal_tflops']:g} TFLOPS"
    if measured is not None:
        text += f" ({measured:.1%} of the measured {peak['measured_tflops']:g})"
    return text


def _gflop(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value / 1e9:.3f} GFLOP"


def _tflops(value: Optional[float]) -> str:
    return "unknown" if value is None else f"{value:g} TFLOPS"


def _gib(value: Optional[int]) -> str:
    return "n/a" if value is None else f"{value / GiB:.2f} GiB"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render a benchmark JSON report as Markdown.")
    parser.add_argument("report", help="JSON written by python -m tests.kit.benchmark.run")
    args = parser.parse_args(argv)
    with open(args.report) as f:
        report = json.load(f)
    if report.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(f"schema version {report.get('schema_version')} is not {SCHEMA_VERSION}")
    sys.stdout.write(render_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Report: the JSON a scenario writes, and the one Markdown template every model line uses.

    python -m tests.kit.benchmark.scenarios.report result.json > result.md
    python -m tests.kit.benchmark.scenarios.report h20.json --against 910b2.json --out h20-vs-910b2.json

``--against`` compares two reports of the same setup measured on different machines (e.g. H20 and Ascend 910B2),
which no single run can do.

The JSON keeps every raw run (every timed step of every rank), so any metric can be recomputed later. The Markdown is
rendered from the JSON and nothing else, and computes nothing: every number in it comes from ``metrics``.
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from ..config.model import DEFAULT_IMPLEMENTATION
from ..metrics.memory import GiB
from ..metrics.summary import comparison as comparison_ratios

SCHEMA_VERSION = 3
SINGLE, SEARCH, SCALING = "single", "search", "scaling"
COMPARISON = "comparison"
# the field a comparison of two machines differs in
MACHINE = "machine"


def build_report(
    measurement: Dict[str, Any],
    repository: Dict[str, Any],
    command: List[str],
    search: Optional[Dict[str, Any]] = None,
    scaling: Optional[Dict[str, Any]] = None,
    comparison: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The report of ``measurement`` (``single.run_single``), the measurement at the maximum of ``search`` or at the
    largest device count of ``scaling``, optionally compared (``compare.run_compare``)."""
    if search is not None and scaling is not None:
        raise ValueError("a report covers one scenario: a search or a scaling curve, not both")
    report = {
        "schema_version": SCHEMA_VERSION,
        "scenario": SEARCH if search is not None else SCALING if scaling is not None else SINGLE,
        "measurement": measurement,
        "environment": measurement["runs"][0]["environment"],
        "repository": repository,
        "command": command,
    }
    if search is not None:
        report["search"] = search
    if scaling is not None:
        # the largest device count is the report's measurement; the others are kept here
        report["scaling"] = {"curve": scaling["curve"], "measurements": scaling["measurements"][:-1]}
    if comparison is not None:
        report[COMPARISON] = comparison
    return report


def against(ours: Dict[str, Any], theirs: Dict[str, Any]) -> Dict[str, Any]:
    """``ours`` with a comparison against ``theirs``: two reports of the same setup measured on different machines.
    Only the machine (and the profiled steps, which are not part of a measurement) may differ."""
    for report in (ours, theirs):
        if report.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"schema version {report.get('schema_version')} is not {SCHEMA_VERSION}")
    if COMPARISON in ours:
        raise ValueError("this report already has a comparison")
    a, b = ours["measurement"]["setup"], theirs["measurement"]["setup"]
    differ = sorted(k for k in a.keys() | b.keys() if k != "profile_steps" and a.get(k) != b.get(k))
    if differ:
        raise ValueError(f"the two reports measure different setups, differing in {differ}")
    return {
        **ours,
        COMPARISON: {
            "field": MACHINE,
            "ours": _machine(ours),
            "theirs": _machine(theirs),
            "measurement": theirs["measurement"],
            "ratios": comparison_ratios(ours["measurement"]["summary"], theirs["measurement"]["summary"]),
            "environment": theirs["environment"],
            "repository": theirs["repository"],
        },
    }


def _machine(report: Dict[str, Any]) -> str:
    env = report["environment"]
    return env.get("device_name", env.get("accelerator", "?"))


# metrics a report does not cover unless its scenario does, listed so that a missing number is never read as zero
NOT_COVERED = {
    "seq_len": "maximum sequence length on one device",
    "micro_batch": "maximum micro-batch on one device",
    SCALING: "data-parallel scaling efficiency",
    COMPARISON: "comparison with another setup (e.g. the plain-PyTorch baseline, another implementation)",
}
SEARCH_NAMES = {"seq_len": "sequence length", "micro_batch": "micro-batch"}


def render_markdown(report: Dict[str, Any]) -> str:
    measurement = report["measurement"]
    setup, summary = measurement["setup"], measurement["summary"]
    time, utilization, memory = summary["time"], summary["utilization"], summary["memory_bytes"]
    flops, peak = measurement["flops"], measurement["peak"]
    env, repo = report["environment"], report["repository"]
    commit = repo.get("commit", "unknown")[:8] + (" (dirty tree)" if repo.get("dirty") else "")
    devices = f"{summary['devices']} × {env.get('device_name', env.get('accelerator', '?'))}"
    implementation = (
        "" if setup["implementation"] == DEFAULT_IMPLEMENTATION else f", {setup['implementation']} implementation"
    )
    lines = [
        f"# Benchmark: {setup['entry']} ({setup['size']}{implementation}), {setup['config']}",
        "",
        "## Summary",
        "",
        f"{_tokens(time['tokens_per_s'])} tokens/s on {devices} "
        f"({_tokens(time['tokens_per_s_per_device'])} per device), "
        f"MFU {_utilization(utilization, 'mfu', peak)}, "
        f"peak reserved memory {_gib(memory.get('reserved_peak'))}. Commit {commit}.",
        "",
    ]
    search = report.get("search")
    if search is not None:
        lines[-1:-1] = [f"Maximum {SEARCH_NAMES[search['field']]} on one device: **{search['max']}**."]
    flags = summary["flags"] + (search or {}).get("flags", [])
    lines += [f"> **Warning:** {flag}" for flag in flags] + ([""] if flags else [])
    lines += [
        "## Configuration",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Entry / size | {setup['entry']} / {setup['size']} ({summary['parameters'] / 1e9:.2f} B parameters) |",
        f"| Implementation | {setup['implementation']} |",
        f"| Plugin and precision | {setup['config']} |",
        f"| Optimizer | {summary['optimizer']} |",
        f"| Devices | {devices} |",
        f"| Sequence length × micro-batch per device | {setup['seq_len']} × {setup['micro_batch']} |",
        f"| Gradient checkpointing | {'on (non-reentrant)' if setup['gradient_checkpointing'] else 'off'} |",
        f"| Steps | {setup['warmup_steps']} warmup + {setup['timed_steps']} timed, "
        f"{len(measurement['runs'])} repeats in fresh processes |",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Definition |",
        "| --- | --- | --- |",
        f"| Throughput | {_tokens(time['tokens_per_s'])} tokens/s | all devices, from the median step time |",
        f"| Throughput per device | {_tokens(time['tokens_per_s_per_device'])} tokens/s | |",
        f"| Samples | {time['samples_per_s']:.2f} /s | |",
        f"| MFU | {_utilization(utilization, 'mfu', peak)} | model FLOPs per token × tokens/s per device ÷ peak; no "
        "recomputation counted |",
        f"| Model TFLOPS per device | {utilization['model_tflops_per_device']:.1f} | |",
        f"| Step time | {_ms(time['step_time_s']['median'])} (p10 {_ms(time['step_time_s']['p10'])}, "
        f"p90 {_ms(time['step_time_s']['p90'])}) | slowest rank per step, device synchronized before and after |",
        f"| Step split | forward + backward {time['forward_backward_share']:.1%}, optimizer "
        f"{time['optimizer_share']:.1%} | of the median step; device synchronized after the backward (which includes "
        "the gradient reduction) |",
        f"| Repeat spread | {summary['repeat_spread']:.1%} | difference of the repeats' median step times |",
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
            f"| HFU | {_utilization(utilization, 'hfu', peak)} | as MFU, with the forward that gradient checkpointing "
            "recomputes |"
        ]
    difference = "no formula" if flops["difference"] is None else f"{flops['difference']:+.2%}"
    lines += [
        "## FLOPs and peak",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Model FLOPs per token | {_gflop(flops['model_flops_per_token'])} (from the {flops['source']}) |",
        f"| Formula | {_gflop(flops['analytic'])} |",
        f"| Counted (`FlopCounterMode`, meta device, {flops['counted_implementation']} implementation) | "
        f"{_gflop(flops['counted'])} ({difference} from the formula) |",
        f"| Hardware FLOPs per token | {_gflop(flops['hardware_flops_per_token'])} |",
        f"| Peak, nominal ({peak['device']}, {peak['dtype']}) | {_tflops(peak['nominal_tflops'])}: "
        f"{peak['nominal_source'] or 'unknown'} |",
        f"| Peak, measured | {_tflops(peak['measured_tflops'])}: {peak['measured_source'] or 'not measured'} |",
        "",
        *_search_section(search),
        *_scaling_section(report.get("scaling"), setup["profile_steps"]),
        *_comparison_section(report.get(COMPARISON), summary, repo),
        "## Environment and reproduction",
        "",
        "| Item | Value |",
        "| --- | --- |",
    ]
    lines += [f"| {key} | {value} |" for key, value in env.items()]
    lines += [f"| commit | {commit} |", "", "```bash", " ".join(report["command"]), "```", "", "## Not covered", ""]
    covered = {search["field"]} if search is not None else set()
    covered |= {key for key in (SCALING, COMPARISON) if key in report}
    missing = [item for key, item in NOT_COVERED.items() if key not in covered]
    if peak["nominal_tflops"] is None:
        missing.insert(0, f"MFU: no peak is known for {peak['device']} in {peak['dtype']}")
    lines += [f"- {item}" for item in missing]
    return "\n".join(lines) + "\n"


def _comparison_section(
    comparison: Optional[Dict[str, Any]], ours: Dict[str, Any], repository: Dict[str, Any]
) -> List[str]:
    lines = ["## Comparison", ""]
    if comparison is None:
        return lines + ["Not compared by this report.", ""]
    field, ratios = comparison["field"], comparison["ratios"]
    theirs = comparison["measurement"]["summary"]
    reserved = ratios["reserved_ratio"]
    lines += [
        f"Two measurements that differ only in `{field}`, on the same {ours['devices']} device(s), by the same "
        f"protocol. {comparison['ours']} trains at {ratios['throughput_ratio']:.2f}× the throughput of "
        f"{comparison['theirs']}"
        + (f" with {reserved:.2f}× its peak reserved memory." if reserved is not None else "."),
        "",
        f"| {field} | Tokens/s per device | MFU | MFU of the measured peak | Step time | Forward + backward | "
        "Peak reserved | Optimizer | Repeat spread |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for value, summary in ((comparison["ours"], ours), (comparison["theirs"], theirs)):
        time = summary["time"]
        lines.append(
            f"| {value} | {_tokens(time['tokens_per_s_per_device'])} | "
            f"{_percent(summary['utilization']['mfu_nominal'])} | "
            f"{_percent(summary['utilization']['mfu_measured'])} | {_ms(time['step_time_s']['median'])} | "
            f"{time['forward_backward_share']:.1%} | {_gib(summary['memory_bytes'].get('reserved_peak'))} | "
            f"{summary['optimizer']} | {summary['repeat_spread']:.1%} |"
        )
    flagged = [f"{comparison['theirs']}: {flag}" for flag in theirs["flags"]]
    other = comparison.get("repository")
    if other is not None and other.get("commit") != repository.get("commit"):
        flagged.append(
            f"the two measurements ran different commits ({repository.get('commit', 'unknown')[:8]} and "
            f"{other.get('commit', 'unknown')[:8]}): the difference is not only the machine"
        )
    return lines + [""] + [f"> **Warning:** {flag}" for flag in flagged] + ([""] if flagged else [])


def _scaling_section(scaling: Optional[Dict[str, Any]], profiled: int) -> List[str]:
    lines = ["## Data-parallel scaling", ""]
    if scaling is None:
        return lines + ["Not measured by this report.", ""]
    points = scaling["curve"]
    lines += [
        f"Weak scaling: every device keeps the micro-batch and sequence length. Efficiency is the throughput per device "
        f"against {points[0]['devices']} device(s). Communication is the share of the step in which communication "
        f"kernels run; exposed, the share in which nothing else runs (the device waits), from {profiled} profiled "
        "steps after the timed ones, the largest over ranks. The other sections describe the largest device count.",
        "",
        "| Devices | Tokens/s | Per device | Efficiency | Step time | MFU | Communication | Exposed | Peak reserved |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p in points:
        lines.append(
            f"| {p['devices']} | {_tokens(p['tokens_per_s'])} | {_tokens(p['tokens_per_s_per_device'])} | "
            f"{p['efficiency']:.1%} | {_ms(p['step_time_s'])} | {_percent(p['mfu_nominal'])} | "
            f"{_percent(p['communication'])} | {_percent(p['exposed_communication'])} | {_gib(p['reserved_peak'])} |"
        )
    flagged = [f"{p['devices']} devices: {flag}" for p in points for flag in p["flags"]]
    return lines + [""] + [f"> **Warning:** {flag}" for flag in flagged] + ([""] if flagged else [])


def _search_section(search: Optional[Dict[str, Any]]) -> List[str]:
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


def _utilization(utilization: Dict[str, Any], name: str, peak: Dict[str, Any]) -> str:
    nominal, measured = utilization[f"{name}_nominal"], utilization[f"{name}_measured"]
    if nominal is None:
        return "n/a (no peak known)"
    text = f"{nominal:.1%} of the nominal {peak['nominal_tflops']:g} TFLOPS"
    if measured is not None:
        text += f" ({measured:.1%} of the measured {peak['measured_tflops']:g})"
    return text


def _tokens(value: float) -> str:
    return f"{value:,.0f}"


def _ms(seconds: float) -> str:
    return f"{seconds * 1e3:.1f} ms"


def _percent(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _gflop(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value / 1e9:.3f} GFLOP"


def _tflops(value: Optional[float]) -> str:
    return "unknown" if value is None else f"{value:g} TFLOPS"


def _gib(value: Optional[int]) -> str:
    return "n/a" if value is None else f"{value / GiB:.2f} GiB"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render a benchmark JSON report as Markdown.")
    parser.add_argument("report", help="JSON written by python -m tests.kit.benchmark.run")
    parser.add_argument("--against", help="another report of the same setup, measured on another machine")
    parser.add_argument("--out", help="with --against: write the combined JSON report here")
    args = parser.parse_args(argv)
    with open(args.report) as f:
        report = json.load(f)
    if report.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(f"schema version {report.get('schema_version')} is not {SCHEMA_VERSION}")
    if args.against:
        with open(args.against) as f:
            report = against(report, json.load(f))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(report, f, indent=2)
    sys.stdout.write(render_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())

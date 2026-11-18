"""Benchmark a model_zoo entry: run one scenario, write the JSON report and its Markdown.

    python -m tests.kit.benchmark.run --entry transformers_llama_for_causal_lm --size 1b --config zero2_bf16
    python -m tests.kit.benchmark.run --entry transformers_qwen3_for_causal_lm --size 1.7b --config zero2_bf16 \\
        --search micro_batch
    python -m tests.kit.benchmark.run --entry transformers_llama_for_causal_lm --size 1b --config zero2_bf16 \\
        --scaling 1,2,4,8 --compare config=pytorch_fsdp2_bf16

Run it on idle devices: other processes on the same devices slow the steps down and count towards the device-side
memory. The report records how much memory was already in use when each run started.
"""

import argparse
import json
import os
import shlex
import sys
from dataclasses import fields
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from tests.kit.numerics.trace import repository_state

from .config.model import DEFAULT_IMPLEMENTATION, PRETRAINED, RANDOM, WEIGHTS, spec_of
from .config.optimizers import OPTIMIZERS
from .config.plugins import CONFIGS
from .config.setup import Setup, parse_field
from .scenarios import report
from .scenarios.compare import run_compare
from .scenarios.scaling import run_scaling
from .scenarios.search import SEARCHED, run_search
from .scenarios.single import DEFAULT_REPEATS, run_single


def default_output(setup: Setup, repository: Dict[str, Any], kind: str) -> str:
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    commit = repository.get("commit", "unknown")[:8]
    implementation = "" if setup.implementation == DEFAULT_IMPLEMENTATION else f"{setup.implementation}-"
    kind = "" if kind == report.SINGLE else f"{kind}-"
    return f"{setup.entry}-{setup.size}-{implementation}{setup.config}-{kind}{date}-{commit}.json"


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    defaults = {f.name: f.default for f in fields(Setup)}
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--entry", required=True, help="model_zoo entry that declares ModelAttribute.benchmark")
    parser.add_argument("--size", help="one of the entry's BenchmarkSpec.models; defaults to the first")
    parser.add_argument(
        "--implementation",
        default=DEFAULT_IMPLEMENTATION,
        help=f"one of the entry's BenchmarkSpec.implementations; defaults to {DEFAULT_IMPLEMENTATION!r}, the model as "
        "built",
    )
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--optimizer", default=defaults["optimizer"], choices=sorted(OPTIMIZERS))
    parser.add_argument(
        "--weights",
        default=defaults["weights"],
        choices=WEIGHTS,
        help="the published weights (default; from the HuggingFace cache) or random ones",
    )
    parser.add_argument("--devices", type=int, default=defaults["devices"], help="one process per device")
    parser.add_argument("--seq-len", type=int, help="defaults to the entry's BenchmarkSpec.seq_len")
    parser.add_argument("--micro-batch", type=int, help="per device; defaults to the entry's BenchmarkSpec")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=defaults["warmup_steps"])
    parser.add_argument("--timed-steps", type=int, default=defaults["timed_steps"])
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--out", help="JSON report; defaults to <entry>-<size>-<config>-<date>-<commit>.json")
    scenario = parser.add_mutually_exclusive_group()
    scenario.add_argument(
        "--search",
        choices=SEARCHED,
        help="find the largest sequence length (at micro-batch 1 unless given) or micro-batch that trains on one "
        "device, then measure there",
    )
    scenario.add_argument(
        "--scaling",
        help="data-parallel scaling: comma-separated device counts, e.g. 1,2,4,8; each keeps the per-device "
        "micro-batch and sequence length",
    )
    parser.add_argument(
        "--compare",
        metavar="FIELD=VALUE",
        help="also measure the same setup with one field changed, e.g. config=pytorch_fsdp2_bf16 or "
        "implementation=default, and compare (at the maximum with --search, the largest device count with --scaling)",
    )
    parser.add_argument("--granularity", type=int, help="search on multiples of this (256 tokens, 1 sample)")
    parser.add_argument("--upper", type=int, help="largest value to try; seq_len defaults to max_position_embeddings")
    args = parser.parse_args(argv)

    spec = spec_of(args.entry)
    if args.search and args.devices != 1:
        parser.error("--search runs on one device; drop --devices")
    if args.scaling and args.devices != 1:
        parser.error("--scaling sets the device counts; drop --devices")
    compare = None
    if args.compare:
        field, separator, text = args.compare.partition("=")
        if not separator:
            parser.error("--compare takes FIELD=VALUE")
        try:
            compare = (field, parse_field(field, text))
        except ValueError as exc:
            parser.error(str(exc))
    # the maximum sequence length is searched at micro-batch 1
    micro_batch = args.micro_batch or (1 if args.search == "seq_len" else spec.micro_batch)
    setup = Setup(
        entry=args.entry,
        size=args.size or spec.default_size,
        config=args.config,
        seq_len=args.seq_len or spec.seq_len,
        micro_batch=micro_batch,
        devices=args.devices,
        implementation=args.implementation,
        optimizer=args.optimizer,
        weights=args.weights,
        gradient_checkpointing=args.gradient_checkpointing,
        warmup_steps=args.warmup_steps,
        timed_steps=args.timed_steps,
    )
    if setup.weights == PRETRAINED and not spec.has_pretrained(setup.size):
        parser.error(f"{args.entry} {setup.size} declares no published weights; add --weights {RANDOM}")
    if args.search and args.search not in spec.searchable:
        parser.error(f"{args.entry} can search {spec.searchable}, not {args.search}")
    if setup.implementation not in spec.implementation_names:
        parser.error(f"--implementation is one of {spec.implementation_names}")
    search = scaling = None
    if args.search:
        found = run_search(setup, args.search, args.granularity, args.upper, repeats=args.repeats)
        search, measurement = found["search"], found["measurement"]
    elif args.scaling:
        scaling = run_scaling(setup, [int(n) for n in args.scaling.split(",")], repeats=args.repeats)
        measurement = scaling["measurements"][-1]
    else:
        measurement = run_single(setup, repeats=args.repeats)
    comparison = run_compare(measurement, *compare, repeats=args.repeats) if compare else None
    command = ["python", "-m", "tests.kit.benchmark.run", *map(shlex.quote, argv)]
    result = report.build_report(measurement, repository_state(), command, search, scaling, comparison)
    out = args.out or default_output(setup, result["repository"], result["scenario"])
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    markdown = report.render_markdown(result)
    with open(os.path.splitext(out)[0] + ".md", "w") as f:
        f.write(markdown)
    print(markdown)
    print(f"wrote {out} and {os.path.splitext(out)[0]}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

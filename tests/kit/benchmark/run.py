"""Benchmark a model_zoo entry: measure one setup in fresh processes, write the JSON report and its Markdown.

    python -m tests.kit.benchmark.run --entry transformers_llama_for_causal_lm --size 1b --config zero2_bf16
    python -m tests.kit.benchmark.run --entry transformers_qwen3_for_causal_lm --size 8b --config zero2_bf16 \\
        --nproc 8 --seq-len 4096 --micro-batch 1 --gradient-checkpointing --out qwen3-8b.json

Each repeat starts its own processes, so a repeat inherits nothing (allocator state, compiled kernels, a fragmented
heap) from the one before. Run it on idle devices: other processes on the same devices slow the steps down and count
towards the device-side memory. The report records how much memory was already in use when the run started.
"""

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from dataclasses import asdict, fields
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.testing import spawn
from tests.kit.numerics.trace import environment_meta, repository_state

from . import measure, report
from .flops import flops_per_token
from .peak import Peak, lookup
from .spec import BenchmarkSpec

DEFAULT_REPEATS = 2


def benchmark_spec(entry: str) -> BenchmarkSpec:
    from tests.kit.model_zoo import model_zoo

    if entry not in model_zoo:
        raise ValueError(f"no model_zoo entry {entry!r}")
    attribute = model_zoo[entry][4]
    spec = attribute.benchmark if attribute is not None else None
    if spec is None:
        raise ValueError(f"{entry} declares no ModelAttribute.benchmark")
    return spec


def run_benchmark(
    setup: measure.Setup,
    nproc: int = 1,
    repeats: int = DEFAULT_REPEATS,
    spec: Optional[BenchmarkSpec] = None,
    command: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Measure ``setup`` ``repeats`` times on ``nproc`` processes and return the report. Call from a process that has
    not touched the device: the measuring processes are spawned, and a context held here would count against them.

    ``spec`` defaults to the entry's ``ModelAttribute.benchmark``; one passed here must be picklable (module-level
    functions), as it is sent to the spawned processes."""
    if repeats < 1:
        raise ValueError("need at least one repeat")
    # on the meta device: no memory, and this process stays off the device
    flops = flops_per_token(
        spec or benchmark_spec(setup.entry), setup.size, setup.seq_len, setup.micro_batch, setup.gradient_checkpointing
    )
    runs, environment = [], None
    for _ in range(repeats):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "run.json")
            spawn(_worker, nproc, setup=setup, spec=spec, path=path)
            with open(path) as f:
                result = json.load(f)
        # recorded by the first repeat's rank 0; the later repeats run on the same machine
        run_environment = result.pop("environment")
        environment = environment or run_environment
        runs.append(result)
    dtype = measure.CONFIGS[setup.config].dtype
    device = environment.get("device_name", environment["accelerator"])
    # an unknown device keeps its name and no TFLOPS: the report then says that MFU is missing and why
    peak = asdict(lookup(device, dtype) or Peak(device, dtype, nominal_tflops=None, nominal_source=""))
    return report.build_report(runs, environment, repository_state(), command or [], flops, peak)


def _worker(rank, world_size, port, setup, spec, path):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)
    spec = spec or benchmark_spec(setup.entry)
    used_at_start = _device_used()
    result = measure.measure(spec, setup)
    used = [None] * world_size
    torch.distributed.all_gather_object(used, used_at_start)
    if rank == 0:
        env = _environment()
        if used_at_start is not None:
            # includes this process's own context; well above that, another process shares the device
            env["device_used_at_start_gib"] = round(max(used) / report.GiB, 2)
        result["environment"] = env
        with open(path, "w") as f:
            json.dump(result, f)
    torch.distributed.destroy_process_group()


def _device_used() -> Optional[int]:
    accelerator = get_accelerator()
    if accelerator.name == "cpu":
        return None
    free, total = getattr(torch, accelerator.name).mem_get_info()
    return total - free


def _environment() -> Dict[str, Any]:
    accelerator = get_accelerator()
    env = environment_meta()
    env.update(colossalai=getattr(colossalai, "__version__", "unknown"), accelerator=accelerator.name)
    if accelerator.name == "cpu":
        env["cpu"] = platform.processor() or platform.machine()
        return env
    env["device_name"] = accelerator.get_device_name(0)
    env["device_count"] = accelerator.device_count()
    # the allocator's settings (e.g. expandable_segments) change how much fits
    env["alloc_conf"] = os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF" if accelerator.name == "cuda" else "PYTORCH_NPU_ALLOC_CONF", ""
    )
    if accelerator.name == "cuda":
        env["cuda"] = torch.version.cuda
        env.update(_nvidia_smi())
    elif accelerator.name == "npu":
        import torch_npu

        env["torch_npu"] = torch_npu.__version__
    return env


def _nvidia_smi() -> Dict[str, str]:
    # the clock and power limits bound the achievable throughput; a lowered limit explains a slow run
    query = {"driver_version": "driver", "clocks.max.sm": "max_sm_clock", "power.limit": "power_limit"}
    try:
        # by UUID: the process's device 0 is any physical GPU under CUDA_VISIBLE_DEVICES, nvidia-smi counts physical ones
        uuid = f"GPU-{torch.cuda.get_device_properties(torch.cuda.current_device()).uuid}"
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(query)}", "--format=csv,noheader", f"--id={uuid}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {}
    return dict(zip(query.values(), (value.strip() for value in out.split(","))))


def default_output(setup: measure.Setup, repository: Dict[str, Any], search: Optional[str] = None) -> str:
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    commit = repository.get("commit", "unknown")[:8]
    kind = f"max-{search}-" if search else ""
    return f"{setup.entry}-{setup.size}-{setup.config}-{kind}{date}-{commit}.json"


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    defaults = {f.name: f.default for f in fields(measure.Setup)}
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--entry", required=True, help="model_zoo entry that declares ModelAttribute.benchmark")
    parser.add_argument("--size", help="one of the entry's BenchmarkSpec.models; defaults to the first")
    parser.add_argument("--config", required=True, choices=sorted(measure.CONFIGS))
    parser.add_argument("--nproc", type=int, default=1, help="processes, one device each")
    parser.add_argument("--seq-len", type=int, help="defaults to the entry's BenchmarkSpec.seq_len")
    parser.add_argument("--micro-batch", type=int, help="per device; defaults to the entry's BenchmarkSpec")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=defaults["warmup_steps"])
    parser.add_argument("--timed-steps", type=int, default=defaults["timed_steps"])
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--out", help="JSON report; defaults to <entry>-<size>-<config>-<date>-<commit>.json")
    parser.add_argument(
        "--search",
        choices=["seq_len", "micro_batch"],
        help="find the largest sequence length (at micro-batch 1 unless given) or micro-batch that trains on one "
        "device, then measure there",
    )
    parser.add_argument("--granularity", type=int, help="search on multiples of this (256 tokens, 1 sample)")
    parser.add_argument("--upper", type=int, help="largest value to try; seq_len defaults to max_position_embeddings")
    args = parser.parse_args(argv)

    spec = benchmark_spec(args.entry)
    if args.search and args.nproc != 1:
        parser.error("--search runs on one device; drop --nproc")
    # the maximum sequence length is searched at micro-batch 1
    micro_batch = args.micro_batch or (1 if args.search == "seq_len" else spec.micro_batch)
    setup = measure.Setup(
        entry=args.entry,
        size=args.size or spec.default_size,
        config=args.config,
        seq_len=args.seq_len or spec.seq_len,
        micro_batch=micro_batch,
        gradient_checkpointing=args.gradient_checkpointing,
        warmup_steps=args.warmup_steps,
        timed_steps=args.timed_steps,
    )
    command = ["python", "-m", "tests.kit.benchmark.run", *map(shlex.quote, argv)]
    if args.search:
        from .search import default_upper, run_search

        upper = args.upper or default_upper(spec, setup.size, args.search)
        result = run_search(setup, args.search, args.granularity, upper, repeats=args.repeats, command=command)
    else:
        result = run_benchmark(setup, nproc=args.nproc, repeats=args.repeats, command=command)
    out = args.out or default_output(setup, result["repository"], args.search)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    markdown = os.path.splitext(out)[0] + ".md"
    with open(markdown, "w") as f:
        f.write(report.render_markdown(result))
    print(report.render_markdown(result))
    print(f"wrote {out} and {markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

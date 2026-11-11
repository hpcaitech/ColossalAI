"""One benchmark run: train a model_zoo entry for warmup and timed steps under one plugin, on every rank of a process
group, and return the step times, throughput and peak memory.

Protocol (the same for every entry, so reports compare):

- Each rank builds one synthetic batch of fixed shape (seeded by rank) and feeds it to every step.
- A step is forward, ``booster.backward``, ``optimizer.step`` and ``optimizer.zero_grad``. The device is synchronized
  right before and right after it, so the time covers the device work, not only the host launching it.
- Warmup steps are discarded. The time of a timed step is the maximum over ranks: the slowest rank sets the pace.
- Peak memory is tracked from the first step on, after the model and optimizer are built and boosted.

Everything device-specific goes through ``colossalai.accelerator``, so the same code runs on CUDA and Ascend NPU.
"""

import math
import resource
import sys
import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecision

from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.booster.plugin.torch_fsdp_plugin import TorchFSDPPlugin
from tests.kit.numerics.capture.plugins import make_optimizer

from .spec import BenchmarkSpec

WARMUP_STEPS = 5
TIMED_STEPS = 20
# profiler range of step i (warmup included): f"{STEP_ANNOTATION}{i}"
STEP_ANNOTATION = "benchmark/step"


@dataclass(frozen=True)
class Config:
    """A plugin and precision to benchmark under. ``plugin_fn`` builds the plugin inside each process; ``dtype`` is
    what the matmuls run in, which selects the device's peak for MFU."""

    plugin_fn: Callable
    dtype: str
    mixed_precision: Optional[str] = None


FSDP_BF16 = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
# The names match the configurations of tests/kit/numerics/acceptance.py, so a model's alignment and benchmark reports
# refer to the same setups.
CONFIGS: Dict[str, Config] = {
    "ddp_fp32": Config(TorchDDPPlugin, "fp32"),
    # autocast: the matmuls run in fp16
    "ddp_fp16": Config(TorchDDPPlugin, "fp16", "fp16"),
    "zero1_bf16": Config(partial(LowLevelZeroPlugin, stage=1, precision="bf16"), "bf16"),
    "zero2_bf16": Config(partial(LowLevelZeroPlugin, stage=2, precision="bf16"), "bf16"),
    "zero2_bf16_offload": Config(partial(LowLevelZeroPlugin, stage=2, precision="bf16", cpu_offload=True), "bf16"),
    "gemini_bf16": Config(partial(GeminiPlugin, precision="bf16"), "bf16"),
    "fsdp_fp32": Config(TorchFSDPPlugin, "fp32"),
    "fsdp_bf16": Config(partial(TorchFSDPPlugin, mixed_precision=FSDP_BF16), "bf16"),
}


@dataclass(frozen=True)
class Setup:
    """What one run measures. Every field is recorded in the report."""

    entry: str
    size: str
    config: str
    seq_len: int
    micro_batch: int
    gradient_checkpointing: bool = False
    warmup_steps: int = WARMUP_STEPS
    timed_steps: int = TIMED_STEPS
    lr: float = 1e-5
    seed: int = 1024

    def __post_init__(self):
        if self.config not in CONFIGS:
            raise ValueError(f"unknown config {self.config!r}; known: {sorted(CONFIGS)}")
        if self.timed_steps < 1 or self.warmup_steps < 0:
            raise ValueError("need at least one timed step and no negative warmup")


def measure(spec: BenchmarkSpec, setup: Setup) -> Dict[str, Any]:
    """Run ``setup`` on every rank of the initialized process group; every rank returns the same result dict.

    Collective: all ranks must call it."""
    if setup.size not in spec.models:
        raise ValueError(f"{setup.entry} has no size {setup.size!r}; sizes: {list(spec.models)}")
    accelerator = get_accelerator()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(setup.seed)
    config = CONFIGS[setup.config]
    plugin = config.plugin_fn()
    booster = Booster(plugin=plugin, mixed_precision=config.mixed_precision)

    model = spec.models[setup.size]()
    if spec.prepare_fn is not None:
        model = spec.prepare_fn(model) or model
    if setup.gradient_checkpointing:
        # non-reentrant: the variant HuggingFace recommends and every plugin of this tool supports
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = make_optimizer(
        plugin, partial(torch.optim.AdamW, lr=setup.lr), [p for p in model.parameters() if p.requires_grad]
    )
    optimizer_name = type(optimizer).__name__
    generator = torch.Generator().manual_seed(setup.seed + rank)
    batch = spec.batch_fn(model, setup.micro_batch, setup.seq_len, generator)
    tokens = spec.tokens_fn(batch)
    model, optimizer, *_ = booster.boost(model, optimizer)
    device = accelerator.get_current_device()
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # on the CPU every operation has finished when it returns; its accelerator has no synchronize() and no allocator
    # statistics
    on_cpu = accelerator.name == "cpu"
    synchronize = (lambda: None) if on_cpu else accelerator.synchronize
    has_memory_stats = not on_cpu
    if has_memory_stats:
        synchronize()
        setup_allocated = accelerator.memory_allocated()
        accelerator.reset_peak_memory_stats()
    times, losses = [], []
    for step in range(setup.warmup_steps + setup.timed_steps):
        synchronize()
        start = time.perf_counter()
        # names each step in a profiler trace, on the host and the device timeline, to check the timing against
        with torch.profiler.record_function(f"{STEP_ANNOTATION}{step}"):
            output = model(**batch)
            loss = spec.loss_fn(output)
            booster.backward(loss, optimizer)
            optimizer.step()
            optimizer.zero_grad()
        synchronize()
        times.append(time.perf_counter() - start)
        losses.append(loss.detach().float().item())

    memory = _memory(accelerator, setup_allocated) if has_memory_stats else {}
    memory["host_rss_peak"] = _host_rss_peak()
    ranks = _gather({"times": times[setup.warmup_steps :], "memory": memory, "tokens": tokens})
    # the slowest rank sets the pace of a step; memory is reported for the rank that uses the most
    step_times = [max(steps) for steps in zip(*(r["times"] for r in ranks))]
    memory = {key: max(r["memory"][key] for r in ranks) for key in memory}
    global_tokens = sum(r["tokens"] for r in ranks)
    return {
        "setup": asdict(setup),
        "world_size": world_size,
        "parameters": parameters,
        "optimizer": optimizer_name,
        "tokens_per_step": global_tokens,
        "samples_per_step": setup.micro_batch * world_size,
        "step_time_s": step_times,
        "memory_bytes": memory,
        "loss": {"first": losses[0], "last": losses[-1], "finite": all(math.isfinite(x) for x in losses)},
    }


def _memory(accelerator, setup_allocated: int) -> Dict[str, int]:
    """Peak allocator statistics since the first step, and an estimate of the device-side peak (``nvidia-smi`` /
    ``npu-smi``): the allocator's reserved peak plus what the process holds outside the allocator (context,
    communication buffers), measured now as device usage minus currently reserved memory. Other processes on the same
    device count too, which is why the device should be idle."""
    accelerator.synchronize()
    reserved_now = accelerator.memory_reserved()
    free, total = getattr(torch, accelerator.name).mem_get_info()
    outside_allocator = max(total - free - reserved_now, 0)
    return {
        "setup_allocated": setup_allocated,
        "allocated_peak": accelerator.max_memory_allocated(),
        "reserved_peak": accelerator.max_memory_reserved(),
        "device_peak": accelerator.max_memory_reserved() + outside_allocator,
        "device_total": total,
    }


def _host_rss_peak() -> int:
    # ru_maxrss is in KiB on Linux and in bytes on macOS
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _gather(value: Any) -> List[Any]:
    # objects, not tensors: works with every backend (gloo, NCCL, HCCL) and keeps Python ints and floats exact
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values

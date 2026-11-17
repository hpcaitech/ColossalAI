"""Layer 2, runner: run one ``Setup`` once, in fresh processes (one per device), and record raw data. No metric is
computed here; ``metrics`` turns the record into numbers.

Protocol (the same for every entry, so reports compare):

- Each rank builds one synthetic batch of fixed shape (seeded by rank) and feeds it to every step.
- A step is forward, backward (``booster.backward`` under a plugin), ``optimizer.step`` and ``optimizer.zero_grad``.
  The device is synchronized before the step, after the backward and after the step, so each time covers the device
  work, not only the host launching it. The time after the backward splits the step into forward + backward
  (including the gradient reduction a plugin runs in its backward hooks) and the optimizer (including the parameter
  gathering a sharded optimizer runs in its step).
- Warmup steps are not recorded. Timed steps are recorded on every rank.
- Peak memory is tracked from the first step on, after the model and optimizer are built and boosted.
- Optionally (``Setup.profile_steps``) a few more steps run under the profiler, after the timed ones and outside both
  timing and memory, recording the device timeline (``timeline.py``).

A run that runs out of device memory, or whose process the host kills (out of host memory), is recorded with that
status rather than raised: for a maximum search, "does not fit" is a result. Any other error is raised.

Everything device-specific goes through ``colossalai.accelerator``, so the same code runs on CUDA and Ascend NPU.
"""

import json
import os
import re
import resource
import sys
import tempfile
import time
from dataclasses import asdict
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import ProcessExitedException, ProcessRaisedException

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.testing import spawn
from tests.kit.numerics.capture.plugins import make_optimizer

from ..config.model import BenchmarkSpec, spec_of
from ..config.plugins import CONFIGS
from ..config.setup import Setup
from . import timeline
from .environment import device_used, environment

COMPLETED = "completed"
DEVICE_OUT_OF_MEMORY = "device_out_of_memory"
HOST_OUT_OF_MEMORY = "host_out_of_memory"
# what CUDA (torch.OutOfMemoryError) and Ascend ("NPU out of memory") raise when an allocation fails
_OUT_OF_MEMORY = re.compile(r"OutOfMemoryError|out of memory", re.IGNORECASE)


def measure(setup: Setup, spec: Optional[BenchmarkSpec] = None) -> Dict[str, Any]:
    """Run ``setup`` once on ``setup.devices`` fresh processes and return its record. Call from a process that has
    not touched the device: a context held here would count against the measured processes.

    ``spec`` defaults to the entry's ``ModelAttribute.benchmark``; one passed here must be picklable (module-level
    functions), as it is sent to the spawned processes.

    The record has ``status`` (``COMPLETED``, ``DEVICE_OUT_OF_MEMORY`` or ``HOST_OUT_OF_MEMORY``) and ``setup``; a
    completed one also ``parameters``, ``optimizer``, ``environment`` (rank 0's) and ``ranks``, one dict per rank:

    - ``tokens``, ``samples``: what the rank trains per step;
    - ``step_s``, ``forward_backward_s``: per timed step, its time and the time to the end of its backward;
    - ``losses``: every step's loss, warmup included;
    - ``memory``: bytes, peaks since the first step (allocator statistics only on a device);
    - ``timeline``: ``timeline.spans_from_trace`` of the profiled steps, or None;
    - ``device_used_at_start``: bytes of the device in use before the model was built, by any process."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "record.json")
        try:
            spawn(_worker, setup.devices, setup=setup, spec=spec, path=path)
        except ProcessRaisedException as exc:
            if not _OUT_OF_MEMORY.search(str(exc)):
                raise
            return {"status": DEVICE_OUT_OF_MEMORY, "reason": "out of device memory", "setup": asdict(setup)}
        except ProcessExitedException as exc:
            # killed by a signal: the host ran out of memory (an offloading plugin) and the kernel stopped it
            return {"status": HOST_OUT_OF_MEMORY, "reason": f"process exited ({exc})", "setup": asdict(setup)}
        with open(path) as f:
            return json.load(f)


def _worker(rank, world_size, port, setup, spec, path):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)
    spec = spec or spec_of(setup.entry)
    used_at_start = device_used()
    record = _train(spec, setup)
    record["device_used_at_start"] = used_at_start
    ranks = [None] * world_size
    # objects, not tensors: works with every backend (gloo, NCCL, HCCL) and keeps Python ints and floats exact
    dist.all_gather_object(ranks, record)
    if rank == 0:
        shared = {key: ranks[0].pop(key) for key in ("parameters", "optimizer")}
        for other in ranks[1:]:
            for key in shared:
                other.pop(key)
        with open(path, "w") as f:
            json.dump(
                {"status": COMPLETED, "setup": asdict(setup), **shared, "environment": environment(), "ranks": ranks},
                f,
            )
    dist.destroy_process_group()


def _train(spec: BenchmarkSpec, setup: Setup) -> Dict[str, Any]:
    """This rank's raw record of ``setup``. Collective: all ranks must call it."""
    accelerator = get_accelerator()
    rank = dist.get_rank()
    torch.manual_seed(setup.seed)
    config = CONFIGS[setup.config]
    device = accelerator.get_current_device()

    model = spec.build(setup.size, setup.implementation)
    if setup.gradient_checkpointing:
        # non-reentrant: the variant HuggingFace recommends and every plugin of this tool supports
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    parameters = sum(p.numel() for p in model.parameters())
    generator = torch.Generator().manual_seed(setup.seed + rank)
    batch = spec.batch_fn(model, setup.micro_batch, setup.seq_len, generator)
    tokens = spec.tokens_fn(batch)
    if config.plugin_fn is not None:
        plugin = config.plugin_fn()
        booster = Booster(plugin=plugin, mixed_precision=config.mixed_precision)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = make_optimizer(plugin, partial(torch.optim.AdamW, lr=setup.lr), trainable)
        optimizer_name = type(optimizer).__name__
        model, optimizer, *_ = booster.boost(model, optimizer)

        def backward(loss):
            booster.backward(loss, optimizer)

    else:
        model = config.wrap_fn(model, device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=setup.lr)
        optimizer_name = type(optimizer).__name__

        def backward(loss):
            loss.backward()

    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # on the CPU every operation has finished when it returns; its accelerator has no synchronize() and no allocator
    # statistics
    on_cpu = accelerator.name == "cpu"
    synchronize = (lambda: None) if on_cpu else accelerator.synchronize
    if not on_cpu:
        synchronize()
        setup_allocated = accelerator.memory_allocated()
        accelerator.reset_peak_memory_stats()
    step_s, forward_backward_s, losses = [], [], []

    def train_step(step: int):
        synchronize()
        start = time.perf_counter()
        # names each step in a profiler trace, on the host and the device timeline
        with torch.profiler.record_function(f"{timeline.STEP_ANNOTATION}{step}"):
            output = model(**batch)
            loss = spec.loss_fn(output)
            backward(loss)
            synchronize()
            backward_end = time.perf_counter()
            optimizer.step()
            optimizer.zero_grad()
        synchronize()
        end = time.perf_counter()
        step_s.append(end - start)
        forward_backward_s.append(backward_end - start)
        losses.append(loss.detach().float().item())

    for step in range(setup.warmup_steps + setup.timed_steps):
        train_step(step)
    memory = _memory(accelerator, setup_allocated) if not on_cpu else {}
    memory["host_rss_peak"] = _host_rss_peak()
    spans = None
    if setup.profile_steps:
        first = setup.warmup_steps + setup.timed_steps
        spans = timeline.profile(accelerator.name, lambda: [train_step(first + i) for i in range(setup.profile_steps)])
    timed = slice(setup.warmup_steps, setup.warmup_steps + setup.timed_steps)
    return {
        "parameters": parameters,
        "optimizer": optimizer_name,
        "tokens": tokens,
        "samples": setup.micro_batch,
        "step_s": step_s[timed],
        "forward_backward_s": forward_backward_s[timed],
        "losses": losses[: setup.warmup_steps + setup.timed_steps],
        "memory": memory,
        "timeline": spans,
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


class NotCompleted(RuntimeError):
    """A run that ran out of device or host memory, which has no metrics."""

    def __init__(self, record: Dict[str, Any]):
        super().__init__(f"{record['setup']} did not complete: {record['reason']}")
        self.record = record


def completed(record: Dict[str, Any]) -> bool:
    return record["status"] == COMPLETED


def require_completed(records: List[Dict[str, Any]]) -> None:
    for record in records:
        if not completed(record):
            raise NotCompleted(record)

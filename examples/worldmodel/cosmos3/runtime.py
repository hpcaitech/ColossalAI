"""Shared configuration and accounting helpers for Cosmos 3 example entry points."""

from __future__ import annotations

import contextlib
import multiprocessing
import random
import socket

import torch
import torch.distributed as dist


def visit_seed(base_seed: int, epoch: int, index: int) -> int:
    """A stable seed for one visit to one sample, independent of which worker drew it.

    splitmix64 over the three inputs.  The point is that the augmentation becomes a property of
    ``(seed, epoch, index)`` rather than of worker scheduling: a resumed run draws the same samples
    in the same order, so it must also draw the same augmentation for each of them.
    """
    mask = (1 << 64) - 1
    value = (int(base_seed) * 0x9E3779B97F4A7C15 + int(epoch) * 0xBF58476D1CE4E5B9 + int(index)) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return value % (1 << 63)


class VisitRandomness:
    """Per-visit seeding that survives ``DataLoader`` workers.

    A worker's RNG cannot be read or restored, so a dataset that lets the transforms draw from it
    replays different augmentations after a resume.  This object instead derives a seed from
    ``(seed, epoch, index)`` and pins the CPU streams to it for the duration of one visit.  The epoch
    lives in a shared value rather than in the sample index: workers outlive an epoch under
    ``persistent_workers`` and only a shared value is visible to all of them.
    """

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._epoch = multiprocessing.Value("q", 0, lock=False)

    def set_epoch(self, epoch: int) -> None:
        self._epoch.value = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)

    def seed_for(self, index: int) -> int:
        return visit_seed(self.seed, self.epoch, int(index))

    @staticmethod
    @contextlib.contextmanager
    def seeded(seed: int):
        """Pin the CPU torch and Python streams to ``seed``, then put them back."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            state = random.getstate()
            random.seed(int(seed))
            try:
                yield
            finally:
                random.setstate(state)


def resolve_attention_backend(requested: str, device_type: str) -> str:
    """Resolve attention independently of the accelerator and communication backend.

    ``cuda`` names the native PyTorch varlen adapter. Availability is checked by its loader;
    requesting an unavailable adapter must raise instead of changing the training kernel.
    """
    if requested not in ("auto", "naive", "npu", "cuda"):
        raise ValueError(f"Unknown attention backend {requested!r}")
    resolved = device_type if requested == "auto" and device_type in ("npu", "cuda") else requested
    if resolved == "auto":
        resolved = "naive"
    if resolved in ("npu", "cuda") and resolved != device_type:
        raise ValueError(f"Attention backend {resolved!r} requires that device, got {device_type!r}")
    return resolved


def resolve_compile_backend(requested: str, device_type: str) -> str:
    if requested == "auto":
        return "npugraph_ex" if device_type == "npu" else "inductor"
    if requested == "npugraph_ex" and device_type != "npu":
        raise ValueError("The npugraph_ex compiler requires an NPU")
    if requested not in ("npugraph_ex", "inductor"):
        raise ValueError(f"Unknown compiler backend {requested!r}")
    return requested


def parallel_topology(plugin, extra=None) -> list[dict]:
    """Record logical groups and their actual hosts once, at startup.

    HSDP groups follow rank order. A shard group fits on one host only when the launcher
    assigned ranks that way; recording hosts makes that assumption observable.

    ``extra`` lets the caller add per-rank facts (local rank, device index, pid) without a second all-gather.
    """
    groups = {}
    for name in ("dp_group", "sp_group", "mixed_dp_group", "shard_group", "replica_group"):
        group = getattr(plugin, name, None)
        groups[name] = None if group is None else dist.get_process_group_ranks(group)
    local = {"rank": dist.get_rank(), "hostname": socket.gethostname(), "groups": groups}
    if extra:
        local.update(extra)
    rows = [None] * dist.get_world_size()
    dist.all_gather_object(rows, local)
    return rows


class AccumulationWindow:
    """Average display MSEs by sample and retain the optimizer's mean objective over microsteps."""

    def __init__(self, count_samples: bool = True):
        self.count_samples = count_samples
        self.samples = self.ar_tokens = self.dm_tokens = self.microsteps = 0
        self.metric_samples = 0
        self.wait_ms = 0.0
        self.loss_sum = None
        self.component_names = ()

    def add(self, samples: int, meta, loss, components: dict, wait_ms: float = 0.0):
        self.microsteps += 1
        if self.count_samples:
            self.samples += samples
            self.ar_tokens += meta.total_ar_tokens
            self.dm_tokens += meta.total_dm_tokens
        self.wait_ms += wait_ms
        names = tuple(components)
        if self.microsteps > 1 and names != self.component_names:
            raise ValueError("Loss components changed inside an accumulation window")
        self.component_names = names
        self.metric_samples += samples
        values = torch.stack([loss.detach(), *(components[name].detach() * samples for name in names)])
        self.loss_sum = values if self.loss_sum is None else self.loss_sum + values

    def loss_values(self) -> dict[str, float]:
        if self.microsteps == 0:
            raise ValueError("Cannot report an empty accumulation window")
        values = self.loss_sum.tolist()
        components = dict(zip(self.component_names, (value / self.metric_samples for value in values[1:])))
        return {"total": sum(components.values()), **components, "objective": values[0] / self.microsteps}


def shutdown_dataloader(iterator) -> None:
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()

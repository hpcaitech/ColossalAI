"""Shared configuration and accounting helpers for Cosmos 3 example entry points."""

from __future__ import annotations

import socket

import torch
import torch.distributed as dist


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
    ``extra`` adds per-rank facts (local rank, device index, pid) without a second all-gather.
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
    """Count independent samples and average loss over every microstep of an optimizer step."""

    def __init__(self, count_samples: bool = True):
        self.count_samples = count_samples
        self.samples = self.ar_tokens = self.dm_tokens = self.microsteps = 0
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
        values = torch.stack([loss.detach(), *(components[name].detach() for name in names)])
        self.loss_sum = values if self.loss_sum is None else self.loss_sum + values

    def loss_values(self) -> dict[str, float]:
        if self.microsteps == 0:
            raise ValueError("Cannot report an empty accumulation window")
        values = (self.loss_sum / self.microsteps).tolist()
        return dict(zip(("total", *self.component_names), values))


def shutdown_dataloader(iterator) -> None:
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()

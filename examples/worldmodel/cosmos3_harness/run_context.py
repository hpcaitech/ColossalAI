"""Identity and launch metadata that make a run reproducible after the fact.

The training loop already records its loss and timing.  This module records the things a person asks about a
week later: which command was run, from which commit, on which hosts and cards, with which checkpoint cadence,
and where the log went.  Every lookup degrades to ``None`` instead of failing a training job.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = [
    "collect_run_identity",
    "collect_rank_identity",
    "enrich_manifest",
    "launcher_env",
    "sanitize_argv",
]

_SECRET_HINTS = ("token", "secret", "password", "passwd", "api_key", "apikey", "access_key")
_REPO_ROOT = Path(__file__).resolve().parents[3]


def sanitize_argv(argv, max_items: int = 512, max_chars: int = 4096):
    """Redact token-like flags before a command line is copied into a report."""
    items = []
    redact_next = False
    for raw in list(argv)[:max_items]:
        value = str(raw)
        if redact_next:
            items.append("<redacted>")
            redact_next = False
            continue
        if value.startswith("-") and "=" in value:
            flag, _, _ = value.partition("=")
            if any(hint in flag.lower().replace("-", "_") for hint in _SECRET_HINTS):
                items.append(flag + "=<redacted>")
                continue
        if value.startswith("-") and any(hint in value.lower().replace("-", "_") for hint in _SECRET_HINTS):
            items.append(value)
            redact_next = True
            continue
        items.append(value[:max_chars])
    return items


def _git() -> Dict[str, Any]:
    result: Dict[str, Any] = {"commit": None, "branch": None, "dirty": None, "dirty_files": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=2, check=False
        )
        if commit.returncode == 0:
            result["commit"] = commit.stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if branch.returncode == 0:
            result["branch"] = branch.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if status.returncode == 0:
            lines = [line for line in status.stdout.splitlines() if line.strip()]
            result["dirty_files"] = len(lines)
            result["dirty"] = bool(lines)
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def launcher_env() -> Dict[str, Any]:
    """The torchrun/Ascend environment as it was at launch, absent values included as ``None``."""
    names = (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "NODE_RANK",
        "NNODES",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
        "ASCEND_RT_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "HCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "OMP_NUM_THREADS",
    )
    result: Dict[str, Any] = {}
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            result[name.lower()] = None
        elif raw.isdigit():
            result[name.lower()] = int(raw)
        else:
            result[name.lower()] = raw
    return result


def _host() -> Dict[str, Any]:
    total_gib = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                total_gib = float(line.split()[1]) / 1024 / 1024
                break
    except (OSError, ValueError, IndexError):
        pass
    return {
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "memory_total_gib": total_gib,
    }


def collect_run_identity(args=None, *, run_name: Optional[str] = None, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Return the identity block merged into ``run.json``."""
    now = time.time()
    chosen_id = run_id
    if chosen_id is None and args is not None:
        chosen_id = getattr(args, "run_id", None)
    if not chosen_id:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
        host = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in socket.gethostname())[:32]
        chosen_id = f"{stamp}-{host}-{os.getpid()}"
    name = run_name or (getattr(args, "run_name", None) if args is not None else None)
    return {
        "run_id": str(chosen_id),
        "run_name": name,
        "started_at": now,
        "run": {
            "argv": sanitize_argv(sys.argv),
            "command": " ".join(sanitize_argv(sys.argv)),
            "cwd": os.getcwd(),
            "python": sys.version.split()[0],
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "git": _git(),
        },
        "launcher": launcher_env(),
        "host": _host(),
    }


def collect_rank_identity(device=None) -> Dict[str, Any]:
    """Per-rank local identity used by the topology table."""
    index = None
    if device is not None:
        index = getattr(device, "index", None)
    if index is None:
        raw = os.environ.get("LOCAL_RANK")
        index = int(raw) if raw and raw.isdigit() else None
    name = None
    if device is not None:
        try:
            import torch

            module = getattr(torch, device.type, None)
            if module is not None and hasattr(module, "get_device_name"):
                name = module.get_device_name(device)
        except Exception:
            name = str(device)
    return {
        "local_rank": index,
        "device_index": index,
        "device_name": name,
        "pid": os.getpid(),
    }


def enrich_manifest(
    manifest: Dict[str, Any],
    *,
    args=None,
    device=None,
    run_name: Optional[str] = None,
    run_id: Optional[str] = None,
    checkpoint_schedule: Optional[Dict[str, Any]] = None,
    log_path: Optional[str] = None,
    heartbeat_seconds: float = 2.0,
    metrics_every: Optional[int] = None,
    log_every: Optional[int] = None,
    monitor_schema: int = 1,
    identity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return ``manifest`` with the monitor's additive identity sections filled in.

    ``identity`` lets rank 0 compute the git/host/argv block once and broadcast it, instead of every rank running
    the same git commands at startup.
    """
    if identity is None:
        identity = collect_run_identity(args, run_name=run_name, run_id=run_id)
    for key, value in identity.items():
        manifest.setdefault(key, value)
    manifest.setdefault("monitor_schema", monitor_schema)
    if checkpoint_schedule is not None:
        manifest.setdefault("checkpoint_schedule", checkpoint_schedule)
    manifest.setdefault("logs", {"path": log_path})
    manifest.setdefault(
        "monitor",
        {
            "schema_version": monitor_schema,
            "heartbeat_seconds": heartbeat_seconds,
            "metrics_every": metrics_every,
            "log_every": log_every,
            "log_path": log_path,
        },
    )
    return manifest

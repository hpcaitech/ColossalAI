"""Small shared contracts: JSON safety, atomic writes, run identity and argument redaction.

Everything here is deliberately dependency-free so the collector and the remote agent can import it on a
node that has neither FastAPI nor the training stack installed.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCHEMA_VERSION = 1

#: Substrings that make an argument value worth redacting in a persisted manifest.
SECRET_HINTS = ("token", "secret", "password", "passwd", "api_key", "apikey", "access_key")

_RUN_ID_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _finite(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None


def json_safe(value: Any) -> Any:
    """Convert a value into something ``json.dumps(..., allow_nan=False)`` accepts.

    Non-finite floats become ``None`` and are reported by the caller's leaf as ``nonfinite``; a NaN that
    reaches a chart is a lie, not a number.  Paths become strings and unknown objects fall back to ``repr``
    so one odd field never turns an entire API response into a 500.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _finite(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            return repr(value)
    return repr(value)


def nonfinite_leaves(value: Any, prefix: str = "") -> List[str]:
    """Names of every non-finite float in a nested mapping, for the ``nonfinite`` warning list."""
    found: List[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        found.append(prefix or "value")
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(nonfinite_leaves(item, f"{prefix}/{key}" if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(nonfinite_leaves(item, f"{prefix}[{index}]"))
    return found


def atomic_write_json(path, payload: Any, *, indent: Optional[int] = None) -> None:
    """Replace ``path`` with a complete JSON document, never exposing a half-written file to a reader."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    text = json.dumps(payload, default=str, allow_nan=False, indent=indent)
    temporary.write_text(text + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sanitize_argv(argv: Iterable[Any], max_items: int = 512, max_chars: int = 4096) -> List[str]:
    """Redact secrets and cap a command line before it becomes part of a run manifest.

    A manifest is meant to be copied into a report or a bug, so a token passed on the command line must not
    travel with it.  The redaction is intentionally conservative: the value after ``--flag`` is replaced
    when the flag name looks sensitive, including ``--flag=value``.
    """
    items: List[str] = []
    redact_next = False
    for raw in list(argv)[:max_items]:
        value = str(raw)
        if redact_next:
            items.append("<redacted>")
            redact_next = False
            continue
        if value.startswith("-") and "=" in value:
            flag, _, _ = value.partition("=")
            if any(hint in flag.lower().replace("-", "_") for hint in SECRET_HINTS):
                items.append(flag + "=<redacted>")
                continue
        if value.startswith("-") and any(hint in value.lower().replace("-", "_") for hint in SECRET_HINTS):
            items.append(value)
            redact_next = True
            continue
        items.append(value[:max_chars])
    return items


def make_run_id(hostname: str, pid: int, now: Optional[float] = None) -> str:
    """Stable-ish identifier for a fresh launch: UTC time, host and pid."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time() if now is None else now))
    host = _RUN_ID_SAFE.sub("-", str(hostname or "host"))[:32].strip("-") or "host"
    return f"{stamp}-{host}-{int(pid):d}"


def clean_run_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = _RUN_ID_SAFE.sub("-", str(value)).strip("-")
    return text[:128] or None


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def launcher_env() -> Dict[str, Any]:
    """The distributed-launch environment a monitor needs, with absent values kept as ``None``.

    torchrun exports ``RANK``/``LOCAL_RANK``/``WORLD_SIZE`` and ``NNODES``/``NODE_RANK``; the Ascend
    container additionally defines the visible devices and HCCL interface.  Recording the values as they
    were at launch is more useful than recomputing them later.
    """
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
            continue
        result[name.lower()] = int(raw) if raw.isdigit() else raw
    return result

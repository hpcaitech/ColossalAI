"""Append-only checkpoint ledger: when a save started, when it finished, and how big it was.

The ledger is written by rank 0 only.  A checkpoint that begins and never ends is a fact worth keeping (the
writer crashed or is still flushing), so the two records are separate lines merged by ``checkpoint_id`` rather
than one record rewritten in place.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

__all__ = ["CheckpointLedger", "measure_checkpoint"]

_INDEX_NAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
_COMPLETE_NAME = "complete.json"


def _scandir_bytes(root: Path, max_files: int) -> Tuple[Optional[int], Optional[int]]:
    total = 0
    count = 0
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if count >= max_files:
                    return None, count
                try:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                        count += 1
                except OSError:
                    continue
    except OSError:
        return None, None
    return total, count


def measure_checkpoint(path, *, max_files: int = 4096) -> Tuple[Optional[int], Optional[int], str]:
    """Return ``(bytes, files, source)`` without walking a huge tree.

    Preference order: the safetensors index's ``metadata.total_size`` (written by Booster), the harness
    generation's ``complete.json.total_bytes``, a single safetensors file, then a bounded ``scandir`` sum.  A
    ledger must never turn a checkpoint into a slow recursive directory walk.
    """
    root = Path(path)
    if not root.exists():
        return None, None, "missing"
    if root.is_file():
        try:
            return root.stat().st_size, 1, "file"
        except OSError:
            return None, None, "missing"
    for name in _INDEX_NAMES:
        index = root / name
        if index.is_file():
            try:
                payload = json.loads(index.read_text(encoding="utf-8"))
                metadata = payload.get("metadata") or {}
                total = metadata.get("total_size")
                weight_map = payload.get("weight_map") or {}
                if isinstance(total, (int, float)):
                    # ColossalAI's checkpoint IO stores total_size in MiB (calculate_tensor_size returns MB),
                    # unlike a HuggingFace index which stores bytes.  These examples always write the former.
                    return int(total * 2**20), len(weight_map) or None, "index"
            except (OSError, ValueError):
                pass
    complete = root / _COMPLETE_NAME
    if complete.is_file():
        try:
            payload = json.loads(complete.read_text(encoding="utf-8"))
            total = payload.get("total_bytes")
            files = payload.get("sizes") or payload.get("files") or {}
            if isinstance(total, (int, float)):
                return int(total), len(files) or None, "complete.json"
        except (OSError, ValueError):
            pass
    single = root / "model.safetensors"
    if single.is_file():
        try:
            return single.stat().st_size, 1, "safetensors"
        except OSError:
            return None, None, "missing"
    actual, actual_files = _scandir_bytes(root, max_files)
    if actual is None:
        return None, actual_files, "truncated-scan"
    return actual, actual_files, "scandir"


class CheckpointLedger:
    """Write ``checkpoints.jsonl`` next to the rank records."""

    def __init__(
        self,
        directory,
        *,
        rank: int = 0,
        run_id: Optional[str] = None,
        world_size: Optional[int] = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "checkpoints.jsonl"
        self.rank = int(rank)
        self.run_id = run_id
        self.world_size = world_size
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._stream = None
        if self.rank == 0:
            self._stream = self.path.open("a", encoding="utf-8", buffering=1)

    def _write(self, payload: Dict[str, Any]) -> None:
        if self._stream is None:
            return
        self._stream.write(json.dumps(payload, default=str, allow_nan=False) + "\n")

    def begin(
        self,
        *,
        step: int,
        kind: str,
        path,
        async_: bool = False,
        format_: Optional[str] = None,
        generation: Optional[str] = None,
        schedule_every: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record the start of a save and return its id."""
        checkpoint_id = "ckpt-" + uuid.uuid4().hex
        payload = {
            "type": "checkpoint",
            "schema": 1,
            "event": "begin",
            "checkpoint_id": checkpoint_id,
            "step": int(step),
            "kind": kind,
            "path": str(path),
            "async": bool(async_),
            "format": format_,
            "generation": generation,
            "schedule_every": schedule_every,
            "started_at": time.time(),
            "world_size": self.world_size,
            "run_id": self.run_id,
        }
        if extra:
            payload.update(extra)
        self._pending[checkpoint_id] = payload
        self._write(payload)
        return checkpoint_id

    def end(
        self,
        checkpoint_id: str,
        *,
        status: str = "complete",
        path=None,
        generation: Optional[str] = None,
        error: Optional[str] = None,
        bytes_: Optional[int] = None,
        files: Optional[int] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        begin = self._pending.pop(checkpoint_id, None)
        now = time.time()
        started = float(begin.get("started_at", now)) if begin else now
        target = path if path is not None else (begin.get("path") if begin else None)
        if bytes_ is None and target is not None and status == "complete":
            bytes_, measured_files, measured_source = measure_checkpoint(target)
            files = files if files is not None else measured_files
            source = source or measured_source
        payload = {
            "type": "checkpoint",
            "schema": 1,
            "event": "end",
            "checkpoint_id": checkpoint_id,
            "step": int(begin.get("step", 0)) if begin else None,
            "status": status,
            "ended_at": now,
            "duration_ms": max(0.0, (now - started) * 1e3),
            "bytes": bytes_,
            "files": files,
            "source": source,
            "path": str(target) if target is not None else None,
            "generation": generation or (begin.get("generation") if begin else None),
            "error": error,
            "run_id": self.run_id,
            "world_size": self.world_size,
        }
        self._write(payload)
        return payload

    def fail(self, checkpoint_id: str, error: str) -> Dict[str, Any]:
        return self.end(checkpoint_id, status="failed", error=error)

    def complete_pending(self, *, status: str = "complete", error: Optional[str] = None) -> None:
        """Close every still-open save; used in the training loop's final ``finally``."""
        for checkpoint_id in list(self._pending):
            self.end(checkpoint_id, status=status, error=error)

    @property
    def pending(self) -> int:
        return len(self._pending)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

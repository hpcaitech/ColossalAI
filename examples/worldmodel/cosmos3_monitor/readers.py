"""Incremental JSON/JSONL readers that tolerate a writer in the middle of a line.

The dashboard may read a file while a training process appends to it, or a mirror may truncate and rewrite
one.  Both are normal: a partial last line is kept for the next read, and a file that became shorter is
re-read from the beginning instead of seeking past its end.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _stat(path: Path) -> Optional[os.stat_result]:
    try:
        return path.stat()
    except OSError:
        return None


def file_age(path) -> Optional[float]:
    info = _stat(Path(path))
    if info is None:
        return None
    return max(0.0, time.time() - info.st_mtime)


class JsonFile:
    """A JSON document cached by ``(mtime_ns, size)``.

    ``error`` is sticky only for the current read: the next successful read clears it, so a monitor that
    starts before ``run.json`` exists does not stay broken.
    """

    def __init__(self, path, default: Any = None) -> None:
        self.path = Path(path)
        self.default = default
        self.value = default
        self.error: Optional[str] = None
        self._signature: Optional[Tuple[int, int]] = None

    def read(self) -> Any:
        info = _stat(self.path)
        if info is None:
            self.error = "missing"
            return self.value
        signature = (info.st_mtime_ns, info.st_size)
        if signature == self._signature:
            return self.value
        try:
            self.value = json.loads(self.path.read_text(encoding="utf-8"))
            self.error = None
        except (OSError, ValueError) as exc:
            # Keep the previous good value: a writer using atomic replace cannot produce a partial document,
            # so this is usually a genuinely unavailable file or a hand edit in progress.
            self.error = f"{type(exc).__name__}: {exc}"
        self._signature = signature
        return self.value

    @property
    def age_s(self) -> Optional[float]:
        return file_age(self.path)


class JsonlTail:
    """Read only the bytes appended since the previous call.

    Args:
        path: File to tail.
        offset: Initial byte offset.  A mirror can persist this and resume without re-parsing history.
        max_line_bytes: A single pathological line larger than this is discarded rather than buffered
            forever; telemetry lines are kilobytes, not megabytes.
    """

    def __init__(self, path, offset: int = 0, max_line_bytes: int = 4 * 1024 * 1024) -> None:
        self.path = Path(path)
        self.offset = max(0, int(offset))
        self.max_line_bytes = int(max_line_bytes)
        self._pending = b""
        self._inode: Optional[int] = None
        self.error: Optional[str] = None
        self.truncated = False

    def _open(self):
        return self.path.open("rb")

    def read_new(self) -> List[Dict[str, Any]]:
        info = _stat(self.path)
        if info is None:
            self.error = "missing"
            return []
        if self._inode is not None and info.st_ino != self._inode:
            self.reset()
        self._inode = info.st_ino
        if info.st_size < self.offset:
            self.reset()
            self.truncated = True
        if info.st_size == self.offset:
            self.error = None
            return []
        try:
            with self._open() as stream:
                stream.seek(self.offset)
                chunk = stream.read(info.st_size - self.offset)
                self.offset = stream.tell()
        except OSError as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return []
        self.error = None
        payload = self._pending + chunk
        lines = payload.split(b"\n")
        self._pending = lines.pop()
        if len(self._pending) > self.max_line_bytes:
            self._pending = b""
        return self._parse(lines)

    def _parse(self, lines: List[bytes]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                records.append(value)
        return records

    def read_last(self, count: int = 1, max_bytes: int = 262144) -> List[Dict[str, Any]]:
        """Parse the last ``count`` records without moving the incremental offset.

        Used by the list view to learn a run's latest step without reading a multi-gigabyte history.
        """
        info = _stat(self.path)
        if info is None or info.st_size == 0:
            return []
        start = max(0, info.st_size - int(max_bytes))
        try:
            with self._open() as stream:
                stream.seek(start)
                chunk = stream.read()
        except OSError:
            return []
        lines = chunk.split(b"\n")
        if start > 0:
            lines = lines[1:]  # first line is probably cut
        records = self._parse(lines)
        return records[-max(1, int(count)) :]

    def reset(self) -> None:
        self.offset = 0
        self._pending = b""
        self._inode = None


def read_jsonl(path, limit: int = 0) -> List[Dict[str, Any]]:
    """Read a whole JSONL file (small files only), optionally stopping after ``limit`` records."""
    result: List[Dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if isinstance(value, dict):
                    result.append(value)
                    if limit and len(result) >= limit:
                        break
    except OSError:
        pass
    return result


def read_json(path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default

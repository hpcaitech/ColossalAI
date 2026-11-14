"""Node-side agent for the SSH mirror: return only the bytes appended since the caller's offsets.

This module is deliberately standard-library-only and Python 3.9 compatible so it can run on a node that has
no FastAPI and no training environment.  It never writes to the training metrics directory.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import time
from pathlib import Path


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _tail(path: Path, offset: int):
    try:
        size = path.stat().st_size
    except OSError:
        return {"offset": 0, "size": 0, "data": ""}
    truncated = size < offset
    if truncated:
        offset = 0
    try:
        with path.open("rb") as stream:
            stream.seek(offset)
            chunk = stream.read(size - offset)
    except OSError:
        return {"offset": offset, "size": size, "data": ""}
    return {
        "offset": size,
        "size": size,
        "data": base64.b64encode(chunk).decode("ascii"),
        "truncated": truncated,
    }


def _processes(entry: str, out: str):
    rows = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            argv = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        argv = [item for item in argv if item]
        if not argv:
            continue
        joined = " ".join(argv)
        if entry and entry not in joined:
            continue
        if out and out not in joined:
            continue
        rank = local_rank = None
        try:
            for item in (proc / "environ").read_bytes().split(b"\0"):
                if item.startswith(b"RANK="):
                    rank = int(item[5:] or -1)
                elif item.startswith(b"LOCAL_RANK="):
                    local_rank = int(item[11:] or -1)
        except (OSError, ValueError):
            pass
        row = {"pid": int(proc.name), "rank": rank, "local_rank": local_rank}
        try:
            stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            row["state"] = stat[0]
        except (OSError, IndexError):
            pass
        try:
            row["wchan"] = (proc / "wchan").read_text().strip() or None
        except OSError:
            row["wchan"] = None
        rows.append(row)
    rows.sort(key=lambda item: (item.get("rank") is None, item.get("rank") or 0, item["pid"]))
    return rows


def collect(metrics_dir: Path, control_dir: Path, offsets: dict, node_rank: int, entry: str, out: str, log_path: str = "") -> dict:
    files = {}
    for pattern in ("heartbeat-*.json", "terminal-*.json"):
        for path in metrics_dir.glob(pattern):
            value = _read_json(path)
            if value is not None:
                files[path.name] = value
    for name in ("run.json", "summary.json"):
        value = _read_json(metrics_dir / name)
        if value is not None:
            files[name] = value
    launcher = None
    for candidate in (control_dir / f"launcher-{node_rank}.json", metrics_dir / f"launcher-{node_rank}.json"):
        if candidate.is_file():
            launcher = _read_json(candidate)
            break

    aggregated = _tail(metrics_dir / "aggregated.jsonl", int(offsets.get("aggregated", 0)))
    checkpoints = _tail(metrics_dir / "checkpoints.jsonl", int(offsets.get("checkpoints", 0)))
    system = {}
    system_offsets = offsets.get("system") or {}
    for path in sorted((metrics_dir / "system").glob("node-*.jsonl")):
        system[path.name] = _tail(path, int(system_offsets.get(path.name, 0)))
    log_chunk = {"offset": 0, "size": 0, "data": ""}
    if log_path:
        log_file = Path(log_path)
        try:
            size = log_file.stat().st_size
            start = max(0, size - 256 * 1024)
            with log_file.open("rb") as stream:
                stream.seek(start)
                chunk = stream.read()
            log_chunk = {"offset": size, "size": size, "data": base64.b64encode(chunk).decode("ascii")}
        except OSError:
            pass
    return {
        "hostname": socket.gethostname(),
        "time": time.time(),
        "files": files,
        "launcher": launcher,
        "aggregated": aggregated,
        "checkpoints": checkpoints,
        "system": system,
        "processes": _processes(entry, out),
        "metrics_dir": str(metrics_dir),
        "log": log_chunk,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--offsets", default="{}", help="JSON object of byte offsets")
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--entry", default="train_sft.py")
    parser.add_argument("--out", default="")
    parser.add_argument("--log-path", default="")
    args = parser.parse_args()
    try:
        offsets = json.loads(args.offsets)
    except ValueError:
        offsets = {}
    print(json.dumps(collect(args.metrics_dir, args.control_dir, offsets, args.node_rank, args.entry, args.out, args.log_path)))


if __name__ == "__main__":
    main()

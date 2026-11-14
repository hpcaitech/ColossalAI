"""SSH mirror for node-local metrics directories.

The shared filesystem is the primary transport: the collector writes into the run directory and the server
reads it directly.  Some runs keep ``--metrics-dir`` on node-local ``/tmp`` instead (the validation cluster did
this deliberately), and this module materialises those files under a stable local cache that the same server
can read with ``--roots``.

The cache is keyed by ``run_id``, not a random temporary directory, and the byte offsets are persisted so a
monitor restart continues instead of re-downloading the whole history.
"""

from __future__ import annotations

import argparse
import base64
import json
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contracts import atomic_write_json


class Mirror:
    def __init__(
        self,
        config: Dict[str, Any],
        *,
        cache_root,
        nodes: Optional[List[str]] = None,
        remote_repo: str,
        metrics_dir: Optional[str] = None,
        control_dir: Optional[str] = None,
        interval: float = 5.0,
        entry: str = "train_sft.py",
        log_path: Optional[str] = None,
    ) -> None:
        self.config = dict(config)
        self.run_id = str(config.get("run_id") or Path(str(config.get("out") or "run")).name)
        self.nodes = list(nodes or config.get("nodes") or [])
        if not self.nodes:
            raise ValueError("mirror needs --nodes or config.nodes")
        self.cache = Path(cache_root).expanduser() / self.run_id
        self.metrics = self.cache / "metrics"
        self.metrics.mkdir(parents=True, exist_ok=True)
        self.remote_repo = remote_repo.rstrip("/")
        self.metrics_dir = metrics_dir or str(config.get("metrics_dir") or "")
        self.control_dir = control_dir or str(config.get("control_dir") or "")
        self.interval = float(interval)
        self.entry = entry
        self.out = str(config.get("out") or "")
        self.log_path = log_path or ""
        self.state_path = self.cache / "mirror-state.json"
        self.state = self._load_state()
        self.errors_path = self.metrics / "mirror-errors.jsonl"
        (self.cache / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        self._stop = threading.Event()

    def _load_state(self) -> Dict[str, Any]:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                value.setdefault("nodes", {})
                return value
        except (OSError, ValueError):
            pass
        return {"nodes": {}}

    def _save_state(self) -> None:
        atomic_write_json(self.state_path, self.state)

    def _node_state(self, node: str) -> Dict[str, Any]:
        return self.state["nodes"].setdefault(node, {"aggregated": 0, "checkpoints": 0, "system": {}})

    def _agent_path(self) -> str:
        return self.remote_repo + "/examples/worldmodel/cosmos3_monitor/remote_agent.py"

    def _poll_node(self, node: str) -> Dict[str, Any]:
        index = self.nodes.index(node)
        state = self._node_state(node)
        offsets = {
            "aggregated": state.get("aggregated", 0),
            "checkpoints": state.get("checkpoints", 0),
            "system": state.get("system") or {},
        }
        remote_command = shlex.join(
            [
                "python3",
                self._agent_path(),
                "--metrics-dir",
                self.metrics_dir,
                "--control-dir",
                self.control_dir,
                "--offsets",
                json.dumps(offsets, separators=(",", ":")),
                "--node-rank",
                str(index),
                "--entry",
                Path(self.entry).name,
                "--out",
                self.out,
                "--log-path",
                self.log_path,
            ]
        )
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", node, remote_command]
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        if result.returncode != 0:
            raise RuntimeError(f"ssh {node} exited {result.returncode}: {(result.stderr or result.stdout).strip()[:300]}")
        return json.loads(result.stdout)

    def poll_once(self) -> None:
        for node in self.nodes:
            try:
                payload = self._poll_node(node)
                self._ingest(node, payload)
            except Exception as exc:
                self._error(node, f"{type(exc).__name__}: {exc}")
        self._save_state()

    def _ingest(self, node: str, payload: Dict[str, Any]) -> None:
        state = self._node_state(node)
        files = payload.get("files") or {}
        run_json = files.get("run.json") if isinstance(files, dict) else None
        if isinstance(run_json, dict):
            incoming = run_json.get("run_id")
            previous = state.get("run_id")
            if incoming and previous and incoming != previous:
                # The directory was reused for a new run.  Offsets into the old append-only files are
                # meaningless; truncate the mirror instead of splicing two runs together.
                for name in ("aggregated.jsonl", "checkpoints.jsonl"):
                    try:
                        (self.metrics / name).unlink()
                    except OSError:
                        pass
                for path in (self.metrics / "system").glob("node-*.jsonl"):
                    try:
                        path.unlink()
                    except OSError:
                        pass
                for pattern in ("heartbeat-*.json", "terminal-*.json", "run.json", "summary.json"):
                    for path in self.metrics.glob(pattern):
                        try:
                            path.unlink()
                        except OSError:
                            pass
                state["aggregated"] = 0
                state["checkpoints"] = 0
                state["system"] = {}
            if incoming:
                state["run_id"] = incoming
        for name, value in files.items():
            target = self.metrics / name
            target.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        log_chunk = payload.get("log")
        if isinstance(log_chunk, dict) and log_chunk.get("data"):
            try:
                (self.cache / "train.log").write_bytes(base64.b64decode(log_chunk["data"]))
            except (OSError, ValueError):
                pass
        launcher = payload.get("launcher")
        if launcher is not None:
            (self.cache / f"launcher-{self.nodes.index(node)}.json").write_text(
                json.dumps(launcher, ensure_ascii=False, allow_nan=False) + "\n"
            )
        self._append_delta(self.metrics / "aggregated.jsonl", payload.get("aggregated"), state, "aggregated")
        self._append_delta(self.metrics / "checkpoints.jsonl", payload.get("checkpoints"), state, "checkpoints")
        for name, chunk in (payload.get("system") or {}).items():
            system_state = state.setdefault("system", {})
            offset = int(system_state.get(name, 0))
            self._append_delta(self.metrics / "system" / name, chunk, system_state, name, offset_key=None)

    def _append_delta(self, path: Path, chunk: Optional[Dict[str, Any]], holder: Dict[str, Any], key: str, offset_key: Optional[str] = None) -> None:
        if not chunk:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        data = base64.b64decode(chunk.get("data") or "")
        mode = "wb" if chunk.get("truncated") else "ab"
        with path.open(mode) as stream:
            stream.write(data)
        if offset_key is None:
            holder[key] = int(chunk.get("offset") or 0)
        else:
            holder[offset_key] = int(chunk.get("offset") or 0)

    def _error(self, node: str, message: str) -> None:
        try:
            with self.errors_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"time": time.time(), "node": node, "error": message}) + "\n")
        except OSError:
            pass

    def run(self, max_polls: int = 0, stop_event: Optional[threading.Event] = None) -> None:
        stop = stop_event or self._stop
        polls = 0
        while not stop.is_set():
            started = time.time()
            self.poll_once()
            polls += 1
            if max_polls and polls >= max_polls:
                break
            stop.wait(max(0.2, self.interval - (time.time() - started)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--remote-repo", required=True)
    parser.add_argument("--nodes", default="")
    parser.add_argument("--metrics-dir", default=None)
    parser.add_argument("--control-dir", default=None)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--entry", default="train_sft.py")
    parser.add_argument("--log-path", default=None, help="Remote stdout log to mirror for the Logs tab.")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    nodes = [item for item in args.nodes.split(",") if item] if args.nodes else None
    mirror = Mirror(
        config,
        cache_root=args.cache_root,
        nodes=nodes,
        remote_repo=args.remote_repo,
        metrics_dir=args.metrics_dir,
        control_dir=args.control_dir,
        interval=args.interval,
        entry=args.entry,
        log_path=args.log_path,
    )
    mirror.run(max_polls=1 if args.once else 0)


if __name__ == "__main__":
    main()

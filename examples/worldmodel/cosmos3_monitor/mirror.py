"""SSH mirror for node-local metrics directories.

The shared filesystem is the primary transport: the collector writes into the run directory and the server
reads it directly.  Some runs keep ``--metrics-dir`` on node-local ``/tmp`` or a node-local data disk instead
(the validation cluster and the 6-node SFT benchmark did this deliberately), and this module materialises those
files under a stable local cache that the same server can read with ``--roots``.

Every node is polled, not just rank 0: each node owns a disjoint slice of ``rank-*.jsonl``, ``heartbeat-*``,
``terminal-*`` and ``system/node-*.jsonl``.  A run on six nodes therefore becomes one cache directory with 48
rank files, 48 heartbeats and six system files.  ``run.json``, ``summary.json``, ``aggregated.jsonl``,
``checkpoints.jsonl`` and the training log are rank-0 products and are taken from the first node only.

The cache is keyed by ``run_id``, not a random temporary directory, and the byte offsets are persisted so a
monitor restart continues instead of re-downloading the whole history.  File content is written only when it
changes, and a heartbeat's remote age is preserved on the local mtime; otherwise a dead training would look
fresh merely because the mirror keeps touching its files.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contracts import atomic_write_json

_PRIMARY_FILES = {"run.json", "summary.json"}


class Mirror:
    def __init__(
        self,
        config: Dict[str, Any],
        *,
        cache_root,
        nodes: Optional[List[str]] = None,
        remote_repo: str = "",
        agent_path: Optional[str] = None,
        metrics_dir: Optional[str] = None,
        control_dir: Optional[str] = None,
        interval: float = 5.0,
        entry: str = "train_sft.py",
        log_path: Optional[str] = None,
        remote_prefix: str = "",
        ssh_args: Optional[List[str]] = None,
        timeout: float = 30.0,
        max_workers: int = 8,
        max_initial_bytes: int = 2 * 1024 * 1024,
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
        self.agent_path = agent_path or (
            self.remote_repo + "/examples/worldmodel/cosmos3_monitor/remote_agent.py" if self.remote_repo else ""
        )
        if not self.agent_path:
            raise ValueError("mirror needs --agent-path or --remote-repo")
        self.metrics_dir = metrics_dir or str(config.get("metrics_dir") or "")
        self.control_dir = control_dir or str(config.get("control_dir") or "")
        self.out = str(config.get("out") or "")
        if not self.control_dir and self.out:
            self.control_dir = str(Path(self.out) / "control")
        self.interval = float(interval)
        self.entry = entry
        self.log_path = log_path or ""
        self.remote_prefix = remote_prefix.strip()
        self.ssh_args = list(ssh_args or ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5"])
        self.timeout = float(timeout)
        self.max_workers = max(1, int(max_workers))
        self.max_initial_bytes = max(0, int(max_initial_bytes))
        self.state_path = self.cache / "mirror-state.json"
        self.state = self._load_state()
        # Keep mirror bookkeeping out of metrics/: the run's age is derived from that directory, and a status
        # file rewritten every poll would make a dead run look perpetually fresh.
        self.errors_path = self.cache / "mirror-errors.jsonl"
        (self.cache / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        self._lock = threading.RLock()
        self._stop = threading.Event()

    # -- state --------------------------------------------------------------------------------------------

    def _load_state(self) -> Dict[str, Any]:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                value.setdefault("nodes", {})
                value.setdefault("files", {})
                return value
        except (OSError, ValueError):
            pass
        return {"nodes": {}, "files": {}}

    def _save_state_locked(self) -> None:
        atomic_write_json(self.state_path, self.state)

    def _node_state(self, node: str) -> Dict[str, Any]:
        return self.state["nodes"].setdefault(
            node,
            {
                "aggregated": 0,
                "checkpoints": 0,
                "system": {},
                "ranks": {},
                "last_ok": None,
                "last_error": None,
            },
        )

    def _reset_cache_locked(self) -> None:
        for pattern in (
            "heartbeat-*.json",
            "terminal-*.json",
            "run.json",
            "summary.json",
            "rank-*.jsonl",
            "aggregated.jsonl",
            "checkpoints.jsonl",
        ):
            for path in self.metrics.glob(pattern):
                try:
                    path.unlink()
                except OSError:
                    pass
        for path in (self.metrics / "system").glob("node-*.jsonl"):
            try:
                path.unlink()
            except OSError:
                pass
        for state in self.state["nodes"].values():
            state["aggregated"] = 0
            state["checkpoints"] = 0
            state["system"] = {}
            state["ranks"] = {}
        self.state["files"] = {}
        self.state["run_id"] = None

    # -- polling ------------------------------------------------------------------------------------------

    def _poll_node(self, node: str) -> Dict[str, Any]:
        index = self.nodes.index(node)
        with self._lock:
            state = self._node_state(node)
            offsets = {
                "aggregated": state.get("aggregated", 0),
                "checkpoints": state.get("checkpoints", 0),
                "system": dict(state.get("system") or {}),
                "ranks": dict(state.get("ranks") or {}),
            }
        agent_command = [
            "python3",
            self.agent_path,
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
            "--max-initial-bytes",
            str(self.max_initial_bytes),
        ]
        if self.remote_prefix:
            agent_command = shlex.split(self.remote_prefix) + agent_command
        remote_command = shlex.join(agent_command)
        command = ["ssh"] + self.ssh_args + [node, remote_command]
        result = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f"ssh {node} exited {result.returncode}: {(result.stderr or result.stdout).strip()[:300]}"
            )
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict):
            raise RuntimeError(f"ssh {node} returned {type(payload).__name__}, not a JSON object")
        return payload

    def poll_once(self) -> None:
        nodes = list(self.nodes)
        workers = min(self.max_workers, max(1, len(nodes)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._poll_node, node): node for node in nodes}
            for future in as_completed(futures):
                node = futures[future]
                try:
                    payload = future.result()
                except Exception as exc:
                    self._error(node, f"{type(exc).__name__}: {exc}")
                    continue
                try:
                    with self._lock:
                        self._ingest_locked(node, payload)
                except Exception as exc:
                    self._error(node, f"ingest: {type(exc).__name__}: {exc}")
        with self._lock:
            self._save_state_locked()

    def _ingest(self, node: str, payload: Dict[str, Any]) -> None:
        """Test/embedding entry point: merge one poll's payload under the state lock."""
        with self._lock:
            self._ingest_locked(node, payload)

    def _ingest_locked(self, node: str, payload: Dict[str, Any]) -> None:
        state = self._node_state(node)
        primary = node == self.nodes[0]
        files = payload.get("files") if isinstance(payload.get("files"), dict) else {}
        file_ages = payload.get("file_ages") if isinstance(payload.get("file_ages"), dict) else {}

        run_json = files.get("run.json") if isinstance(files, dict) else None
        incoming = run_json.get("run_id") if isinstance(run_json, dict) else None
        previous = self.state.get("run_id")
        if incoming and previous and incoming != previous:
            # The directory was reused for a new run.  Offsets into the old append-only files are meaningless;
            # truncate the mirror instead of splicing two runs together.
            self._reset_cache_locked()
        if incoming:
            self.state["run_id"] = incoming
        state["hostname"] = payload.get("hostname") or state.get("hostname")
        state["last_ok"] = time.time()
        state["last_error"] = None
        state["ranks_seen"] = sorted((payload.get("ranks") or {}).keys())
        state["system_seen"] = sorted((payload.get("system") or {}).keys())

        for name, value in files.items():
            if name in _PRIMARY_FILES and not primary:
                # Only rank 0 owns run.json/summary.json.  Another node may have a stale copy after a directory
                # reuse; taking the last poll's copy would flip the manifest back and forth.
                continue
            self._write_file_locked(name, value, file_ages.get(name))

        if primary:
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
        if primary:
            self._append_delta(self.metrics / "aggregated.jsonl", payload.get("aggregated"), state, "aggregated")
            self._append_delta(self.metrics / "checkpoints.jsonl", payload.get("checkpoints"), state, "checkpoints")
        for name, chunk in (payload.get("system") or {}).items():
            system_state = state.setdefault("system", {})
            self._append_delta(self.metrics / "system" / name, chunk, system_state, name, offset_key=None)
        for name, chunk in (payload.get("ranks") or {}).items():
            rank_state = state.setdefault("ranks", {})
            self._append_delta(self.metrics / name, chunk, rank_state, name, offset_key=None)

    def _write_file_locked(self, name: str, value: Any, age_s: Any = None) -> None:
        text = json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        hashes = self.state.setdefault("files", {})
        if hashes.get(name) == digest:
            return
        target = self.metrics / name
        # Atomic replace: the server must never parse a half-written heartbeat, and a rename also updates the
        # directory mtime, which is the run's liveness signal when there is no aggregate record yet.
        temporary = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, target)
        hashes[name] = digest
        if age_s is not None:
            try:
                stamp = time.time() - max(0.0, float(age_s))
                os.utime(target, (stamp, stamp))
            except (OSError, TypeError, ValueError):
                pass

    def _append_delta(
        self,
        path: Path,
        chunk: Optional[Dict[str, Any]],
        holder: Dict[str, Any],
        key: str,
        offset_key: Optional[str] = None,
    ) -> None:
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
        with self._lock:
            state = self._node_state(node)
            state["last_error"] = message
            state["last_error_time"] = time.time()
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
    parser.add_argument("config", nargs="?", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--remote-repo", default="")
    parser.add_argument("--agent-path", default=None)
    parser.add_argument("--nodes", default="")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--metrics-dir", default=None)
    parser.add_argument("--control-dir", default=None)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--entry", default="train_sft.py")
    parser.add_argument("--log-path", default=None, help="Remote stdout log to mirror for the Logs tab.")
    parser.add_argument("--remote-prefix", default="", help="Prefix for the remote command, e.g. 'docker exec c'.")
    parser.add_argument("--ssh-arg", action="append", default=[], help="Extra ssh option, repeatable.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-initial-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    nodes = [item for item in args.nodes.split(",") if item] if args.nodes else None
    if nodes:
        config["nodes"] = nodes
    if args.run_id:
        config["run_id"] = args.run_id
    if args.out:
        config["out"] = args.out
    if args.metrics_dir:
        config["metrics_dir"] = args.metrics_dir
    if args.control_dir:
        config["control_dir"] = args.control_dir
    mirror = Mirror(
        config,
        cache_root=args.cache_root,
        nodes=nodes,
        remote_repo=args.remote_repo,
        agent_path=args.agent_path,
        metrics_dir=args.metrics_dir,
        control_dir=args.control_dir,
        interval=args.interval,
        entry=args.entry,
        log_path=args.log_path,
        remote_prefix=args.remote_prefix,
        ssh_args=["-o", "BatchMode=yes", "-o", "ConnectTimeout=5"] + list(args.ssh_arg),
        timeout=args.timeout,
        max_workers=args.max_workers,
        max_initial_bytes=args.max_initial_bytes,
    )
    mirror.run(max_polls=1 if args.once else 0)


if __name__ == "__main__":
    main()

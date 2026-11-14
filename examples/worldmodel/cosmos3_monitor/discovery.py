"""Find run directories under configured roots and give each one a URL-safe identity.

Three layouts are understood, because the repository's two training entries and the older validation
worktree write different things:

1. ``<run>/metrics/run.json`` plus ``rank-*.jsonl`` -- the current SFT and harness layout.
2. ``<run>/run.json`` -- an older layout that kept the manifest at the top.
3. ``<run>/config.json`` plus ``<run>/metrics/`` -- the validation cluster launcher and the SSH mirror.

A directory is a run when one of those markers exists.  Discovering a run prunes the walk below it, so a
large checkpoint tree is never traversed.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .readers import JsonFile, file_age

_SKIP_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    "site-packages",
    "dist-packages",
}


def token_for(path) -> str:
    """Short, stable, URL-safe token for a directory.

    The client only ever sees this token; the real path stays in the server's store.  That keeps a crafted
    request from asking the server to read an arbitrary file.
    """
    digest = hashlib.sha1(str(Path(path).resolve()).encode("utf-8")).hexdigest()
    return digest[:12]


@dataclass
class RunRef:
    path: Path
    metrics_dir: Path
    layout: str
    manifest_path: Optional[Path] = None
    config_path: Optional[Path] = None
    token: str = ""
    run_id: Optional[str] = None
    first_seen: float = 0.0
    last_seen: float = 0.0
    errors: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.token:
            self.token = token_for(self.path)

    @property
    def display_name(self) -> str:
        return self.run_id or self.path.name

    def file(self, name: str) -> Path:
        return self.metrics_dir / name

    @property
    def age_s(self) -> Optional[float]:
        return file_age(self.metrics_dir)

    def signature(self) -> tuple:
        """Cheap change detector for the run list: manifest plus metrics directory mtime."""
        try:
            metrics = self.metrics_dir.stat()
            manifest = self.manifest_path.stat() if self.manifest_path and self.manifest_path.exists() else None
            return (
                metrics.st_mtime_ns,
                manifest.st_mtime_ns if manifest else 0,
            )
        except OSError:
            return (0, 0)


def _detect(path: Path) -> Optional[RunRef]:
    metrics = path / "metrics"
    run_json = metrics / "run.json"
    if run_json.is_file():
        return RunRef(path=path, metrics_dir=metrics, layout="metrics", manifest_path=run_json)
    legacy = path / "run.json"
    if legacy.is_file():
        return RunRef(path=path, metrics_dir=metrics if metrics.is_dir() else path, layout="legacy", manifest_path=legacy)
    config = path / "config.json"
    if config.is_file() and metrics.is_dir():
        return RunRef(path=path, metrics_dir=metrics, layout="validation", config_path=config)
    if metrics.is_dir() and (
        (metrics / "aggregated.jsonl").is_file()
        or any(metrics.glob("rank-*.jsonl"))
        or any(metrics.glob("heartbeat-*.json"))
    ):
        return RunRef(path=path, metrics_dir=metrics, layout="metrics-only")
    return None


def discover(
    roots: Iterable[Any],
    *,
    depth: int = 3,
    max_runs: int = 2000,
    include_hidden: bool = False,
) -> List[RunRef]:
    """Walk each root once and return every run directory found.

    Args:
        roots: Directories to scan.  A root that is itself a run is returned directly.
        depth: How many directory levels below a root to inspect.  Runs are pruned from the walk.
        max_runs: Hard cap so a misconfigured root (for example ``/``) cannot freeze the scanner.
        include_hidden: Whether to descend into dot-directories other than the known cache names.
    """
    found: List[RunRef] = []
    seen = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        if not root.exists():
            continue
        direct = _detect(root)
        if direct is not None:
            if direct.token not in seen:
                seen.add(direct.token)
                found.append(direct)
            continue
        if not root.is_dir():
            continue
        base_depth = len(root.parts)
        for current, dirnames, _filenames in os.walk(root, followlinks=False):
            if len(found) >= max_runs:
                return found
            current_path = Path(current)
            level = len(current_path.parts) - base_depth
            if level >= depth:
                dirnames[:] = []
                continue
            dirnames[:] = sorted(
                name
                for name in dirnames
                if name not in _SKIP_NAMES
                and not name.startswith(".incomplete-")
                and (include_hidden or not name.startswith("."))
            )
            ref = _detect(current_path)
            if ref is not None:
                if ref.token not in seen:
                    seen.add(ref.token)
                    found.append(ref)
                dirnames[:] = []
    return found


def load_manifest(ref: RunRef) -> Dict[str, Any]:
    """Return the most complete manifest available for a run.

    ``run.json`` wins; a validation ``config.json`` supplies the launcher fields and CLI arguments when no
    manifest was mirrored.  The result is a plain dict so callers never depend on either file's exact shape.
    """
    if ref.manifest_path is not None and ref.manifest_path.is_file():
        value = JsonFile(ref.manifest_path).read()
        if isinstance(value, dict):
            return dict(value)
    if ref.config_path is not None and ref.config_path.is_file():
        value = JsonFile(ref.config_path).read()
        if isinstance(value, dict):
            manifest = dict(value)
            manifest.setdefault("config", dict(value))
            manifest.setdefault("args", _args_dict(value.get("args")))
            return manifest
    return {}


def _args_dict(args: Any) -> Dict[str, Any]:
    """Turn a CLI argument list into a best-effort mapping for the config table."""
    result: Dict[str, Any] = {}
    if not isinstance(args, list):
        return result
    index = 0
    while index < len(args):
        item = str(args[index])
        if item.startswith("--"):
            key = item[2:].replace("-", "_")
            if index + 1 < len(args) and not str(args[index + 1]).startswith("--"):
                result[key] = args[index + 1]
                index += 2
                continue
            result[key] = True
        index += 1
    return result


def world_size(manifest: Dict[str, Any], ref: Optional[RunRef] = None) -> int:
    for key in ("world_size", "world"):
        value = manifest.get(key)
        if isinstance(value, int) and value > 0:
            return value
    launcher = manifest.get("launcher") or {}
    if isinstance(launcher, dict):
        value = launcher.get("world_size")
        if isinstance(value, int) and value > 0:
            return value
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else manifest
    nodes = config.get("nodes")
    devices = config.get("devices")
    if isinstance(nodes, list) and nodes:
        per_node = len(devices) if isinstance(devices, list) and devices else 8
        return len(nodes) * per_node
    topology = manifest.get("topology")
    if isinstance(topology, list) and topology:
        return len(topology)
    return 0


def max_steps(manifest: Dict[str, Any]) -> Optional[int]:
    args = manifest.get("args")
    candidates: List[Any] = []
    if isinstance(args, dict):
        candidates.extend([args.get("max_iter"), args.get("steps")])
    elif isinstance(args, list):
        parsed = _args_dict(args)
        candidates.extend([parsed.get("max_iter"), parsed.get("steps")])
    candidates.append(manifest.get("max_iter"))
    for value in candidates:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return None


def derive_topology(manifest: Dict[str, Any], world: int) -> List[Dict[str, Any]]:
    """Return ``rank -> host`` rows, using the manifest's topology or the launcher config.

    A mirrored validation run often has no ``run.json``; its ``config.json`` names the nodes and devices, and
    torchrun assigns contiguous ranks per node.  The rows carry ``derived=True`` so the UI does not present
    an inference as a measurement.
    """
    topology = manifest.get("topology")
    if isinstance(topology, list) and topology:
        return [dict(row) for row in topology if isinstance(row, dict)]
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else manifest
    nodes = config.get("nodes")
    devices = config.get("devices")
    if not isinstance(nodes, list) or not nodes:
        return []
    per_node = len(devices) if isinstance(devices, list) and devices else 8
    rows = []
    for rank in range(world or len(nodes) * per_node):
        rows.append(
            {
                "rank": rank,
                "hostname": nodes[rank // per_node] if rank // per_node < len(nodes) else None,
                "local_rank": rank % per_node,
                "derived": True,
            }
        )
    return rows

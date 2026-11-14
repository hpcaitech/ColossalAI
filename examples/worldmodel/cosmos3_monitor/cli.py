"""Command line entry points for the monitor.

Examples::

    python -m examples.worldmodel.cosmos3_monitor check --roots /mnt/disk0/runs
    python -m examples.worldmodel.cosmos3_monitor serve --roots /mnt/disk0/runs --port 18790
    python -m examples.worldmodel.cosmos3_monitor collect --run-dir /path/to/run --interval 5
    python -m examples.worldmodel.cosmos3_monitor mirror run.json --cache-root ~/.cache/cosmos3_monitor/mirror \\
        --remote-repo /mnt/disk0/jiangzs/nzf/ColossalAI --nodes node5,node7 --metrics-dir /tmp/run/metrics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional


def _default_state_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return str(Path(base) / "cosmos3_monitor")


def _roots(args) -> List[str]:
    roots = list(args.roots or [])
    if not roots and args.run_dir:
        roots = [args.run_dir]
    if not roots:
        roots = ["."]
    return roots


def command_check(args) -> int:
    from .discovery import discover
    from .state import RunState

    roots = _roots(args)
    refs = discover(roots, depth=args.depth, max_runs=args.max_runs)
    rows = []
    for ref in refs:
        state = RunState(ref, stale_seconds=args.stale_seconds, stall_seconds=args.stall_seconds)
        state.refresh()
        snapshot = state.snapshot()
        rows.append(
            {
                "run_id": snapshot["run_id"],
                "token": snapshot["token"],
                "dir": snapshot["dir"],
                "layout": snapshot["layout"],
                "state": snapshot["state"],
                "step": snapshot["step"],
                "max_steps": snapshot["max_steps"],
                "world": snapshot["world"],
                "age_s": round(snapshot["age_s"], 1),
                "issues": [item.get("kind") for item in snapshot["issues"]],
                "checkpoints": snapshot["checkpoints"]["count"],
            }
        )
    print(json.dumps({"roots": roots, "runs": rows}, ensure_ascii=False, indent=2))
    return 0


def command_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: pip install 'fastapi>=0.110' 'uvicorn==0.29.0'", file=sys.stderr)
        return 2
    from .server import create_app
    from .store import RunStore

    roots = _roots(args)
    store = RunStore(
        roots,
        state_dir=args.state_dir,
        scan_interval=args.scan_interval,
        refresh_interval=args.scan_interval,
        stale_seconds=args.stale_seconds,
        stall_seconds=args.stall_seconds,
        depth=args.depth,
        max_runs=args.max_runs,
    )
    app = create_app(store)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"warning: binding {args.host}; this dashboard has no authentication. Prefer an SSH tunnel to "
            "127.0.0.1.",
            file=sys.stderr,
        )
    print(f"cosmos3 monitor: roots={roots} state_dir={args.state_dir} http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, workers=1)
    return 0


def command_collect(args) -> int:
    from .collector import run_collector

    run_collector(
        args.run_dir,
        metrics_dir=args.metrics_dir,
        interval=args.interval,
        node_rank=args.node_rank,
        disk_paths=args.disk_path,
        entry=args.entry,
        nice=args.nice,
        max_samples=args.max_samples,
    )
    return 0


def command_mirror(args) -> int:
    from .mirror import Mirror

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
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
    return 0


def command_export(args) -> int:
    """Offline CSV export, useful when the server cannot reach the run directory."""
    from .discovery import discover
    from .series import SeriesStore
    from .state import RunState

    refs = discover([args.run_dir], depth=1)
    if not refs:
        print(f"no run under {args.run_dir}", file=sys.stderr)
        return 1
    state = RunState(refs[0], stale_seconds=args.stale_seconds, stall_seconds=args.stall_seconds)
    state.refresh()
    keys = [item.strip() for item in args.keys.split(",") if item.strip()]
    data = state.series.query(keys, max_points=200000)
    import csv

    writer = csv.writer(sys.stdout)
    names = list(data)
    writer.writerow(["step"] + names)
    length = max((len(data[name]["x"]) for name in names), default=0)
    for index in range(length):
        step = data[names[0]]["x"][index] if index < len(data[names[0]]["x"]) else ""
        writer.writerow([step] + [data[name]["y"][index] if index < len(data[name]["y"]) else "" for name in names])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cosmos3_monitor", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--roots", nargs="+", help="directories to scan for run folders")
        p.add_argument("--run-dir", help="a single run directory")
        p.add_argument("--depth", type=int, default=3)
        p.add_argument("--max-runs", type=int, default=2000)
        p.add_argument("--stale-seconds", type=float, default=15.0)
        p.add_argument("--stall-seconds", type=float, default=300.0)

    check = sub.add_parser("check", help="scan roots and print a JSON summary")
    common(check)
    check.set_defaults(func=command_check)

    serve = sub.add_parser("serve", help="start the dashboard")
    common(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=18790)
    serve.add_argument("--state-dir", default=_default_state_dir())
    serve.add_argument("--scan-interval", type=float, default=2.0)
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=command_serve)

    collect = sub.add_parser("collect", help="run the per-node system collector")
    collect.add_argument("--run-dir", required=True)
    collect.add_argument("--metrics-dir", default=None)
    collect.add_argument("--interval", type=float, default=5.0)
    collect.add_argument("--node-rank", type=int, default=0)
    collect.add_argument("--disk-path", action="append", default=[])
    collect.add_argument("--entry", default="train_sft.py")
    collect.add_argument("--nice", type=int, default=10)
    collect.add_argument("--max-samples", type=int, default=0)
    collect.set_defaults(func=command_collect)

    mirror = sub.add_parser("mirror", help="mirror node-local metrics over SSH")
    mirror.add_argument("config", help="run config JSON with run_id/out/nodes")
    mirror.add_argument("--cache-root", required=True)
    mirror.add_argument("--remote-repo", required=True)
    mirror.add_argument("--nodes", default="")
    mirror.add_argument("--metrics-dir", default=None)
    mirror.add_argument("--control-dir", default=None)
    mirror.add_argument("--interval", type=float, default=5.0)
    mirror.add_argument("--entry", default="train_sft.py")
    mirror.add_argument("--log-path", default=None, help="Remote stdout log to mirror for the Logs tab.")
    mirror.add_argument("--once", action="store_true")
    mirror.set_defaults(func=command_mirror)

    export = sub.add_parser("export", help="export one run's series as CSV")
    export.add_argument("--run-dir", required=True)
    export.add_argument("--keys", default="loss/total,perf/step_s,perf/samples_per_s")
    export.add_argument("--stale-seconds", type=float, default=15.0)
    export.add_argument("--stall-seconds", type=float, default=300.0)
    export.set_defaults(func=command_export)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

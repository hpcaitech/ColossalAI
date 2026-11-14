"""FastAPI application: a read-only REST/SSE API over :class:`~.store.RunStore` plus the SPA.

The server has one worker and one store.  Every response is JSON-safe; a run that is being written by a
training process can only produce a stale or partial snapshot, never a 500.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import MONITOR_VERSION
from .contracts import json_safe, nonfinite_leaves
from .store import RunStore

DEFAULT_KEYS = [
    "loss/total",
    "loss/vision",
    "loss/action",
    "optim/lr",
    "optim/grad_norm",
    "perf/step_s",
    "perf/samples_per_s",
    "perf/mfu",
    "perf/skew",
    "mem/device_used_gib",
    "mem/peak_reserved_gib",
    "mem/allocated_gib",
    "data/samples",
    "data/wait_s",
    "data/vae_s",
    "data/epoch",
]


def _envelope(data: Any, *, updated_at: Optional[float] = None, warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    payload = json_safe(data)
    now = time.time()
    return {
        "data": payload,
        "meta": {
            "updated_at": updated_at if updated_at is not None else now,
            "served_at": now,
            "warnings": warnings or [],
            "nonfinite": nonfinite_leaves(data),
        },
    }


def _keys(raw: Optional[str]) -> List[str]:
    if not raw:
        return list(DEFAULT_KEYS)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or list(DEFAULT_KEYS)


def _require(store: RunStore, token: str):
    state = store.get(token)
    if state is None:
        raise HTTPException(status_code=404, detail=f"unknown run {token!r}")
    return state


def create_app(store: RunStore) -> FastAPI:
    static_dir = Path(__file__).resolve().parent / "static"
    app = FastAPI(title="Cosmos3 training monitor", version=MONITOR_VERSION, docs_url="/api/docs", redoc_url=None)

    @app.on_event("startup")
    async def _startup() -> None:
        store.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        store.stop()

    @app.get("/api/health")
    def health():
        return _envelope(store.health())

    @app.get("/api/runs")
    def runs(
        state: Optional[str] = None,
        q: str = "",
        sort: str = "updated",
        limit: int = Query(500, ge=1, le=5000),
        offset: int = Query(0, ge=0),
    ):
        rows = store.runs(state=state, query=q, sort=sort)
        page = rows[offset : offset + limit]
        return _envelope({"runs": page, "total": len(rows)})

    @app.get("/api/runs/{token}")
    def run_detail(token: str, include: str = ""):
        _require(store, token)
        wants = {item.strip() for item in include.split(",") if item.strip()}
        snapshot = store.snapshot(token, include_manifest="manifest" in wants)
        if snapshot is None:
            raise HTTPException(status_code=404, detail=f"unknown run {token!r}")
        if "events" in wants:
            snapshot["events"] = store.events(token, limit=200)
        if "system" in wants:
            snapshot["system"] = store.system(token)
        return _envelope(snapshot, updated_at=snapshot.get("updated_at"))

    @app.get("/api/runs/{token}/manifest")
    def manifest(token: str):
        _require(store, token)
        value = store.manifest(token) or {}
        return _envelope(value)

    @app.get("/api/runs/{token}/series")
    def series(
        token: str,
        keys: Optional[str] = None,
        x: str = "step",
        max_points: int = Query(5000, ge=10, le=200000),
        from_step: Optional[float] = None,
        to_step: Optional[float] = None,
    ):
        _require(store, token)
        data = store.series(
            token,
            _keys(keys),
            x_axis="time" if x == "time" else "step",
            max_points=max_points,
            from_step=from_step,
            to_step=to_step,
        )
        if data is None:
            raise HTTPException(status_code=404, detail=f"unknown run {token!r}")
        return _envelope({"series": data, "x": x, "max_points": max_points})

    @app.get("/api/runs/{token}/records")
    def records(
        token: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=2000),
        order: str = "asc",
    ):
        _require(store, token)
        data = store.records(token, offset=offset, limit=limit, order=order)
        return _envelope(data)

    @app.get("/api/runs/{token}/events")
    def events(
        token: str,
        severity: Optional[str] = None,
        kind: Optional[str] = None,
        limit: int = Query(200, ge=1, le=2000),
    ):
        _require(store, token)
        data = store.events(token, severity=severity, kind=kind, limit=limit)
        return _envelope({"events": data or []})

    @app.get("/api/runs/{token}/checkpoints")
    def checkpoints(token: str):
        _require(store, token)
        data = store.checkpoints(token)
        return _envelope(data)

    @app.get("/api/runs/{token}/system")
    def system(token: str, host: Optional[str] = None):
        _require(store, token)
        data = store.system(token, host=host)
        return _envelope(data)

    @app.get("/api/runs/{token}/steps/{step}")
    def step_detail(token: str, step: int):
        _require(store, token)
        data = store.step_detail(token, step)
        if data is None:
            raise HTTPException(status_code=404, detail=f"step {step} is not in this run's retained records")
        return _envelope(data)

    @app.get("/api/runs/{token}/ranks")
    def ranks(token: str):
        _require(store, token)
        data = store.ranks(token)
        return _envelope(data)

    @app.get("/api/runs/{token}/alerts")
    def alerts(token: str):
        state = _require(store, token)
        return _envelope(store.alerts.view(state.ref.token))

    @app.post("/api/runs/{token}/alerts/{alert_id}/ack")
    def ack(token: str, alert_id: str):
        state = _require(store, token)
        ok = store.alerts.ack(state.ref.token, alert_id)
        return _envelope({"acked": ok, "alert_id": alert_id})

    @app.get("/api/runs/{token}/log")
    def log(token: str, tail: int = Query(500, ge=1, le=5000), offset: int = 0):
        _require(store, token)
        path = store.log_path(token)
        if path is None:
            raise HTTPException(status_code=404, detail="this run did not record an allowed log path")
        try:
            size = path.stat().st_size
            start = max(0, size - 2 * 1024 * 1024)
            with path.open("rb") as stream:
                stream.seek(start)
                chunk = stream.read()
        except OSError as exc:
            raise HTTPException(status_code=404, detail=f"cannot read log: {exc}")
        lines = chunk.decode("utf-8", errors="replace").splitlines()
        return _envelope({"lines": lines[-tail:], "size": size, "path": str(path)})

    @app.get("/api/runs/{token}/stream")
    async def stream(token: str, request: Request):
        _require(store, token)

        async def generate():
            last_marker = None
            while True:
                if await request.is_disconnected():
                    break
                snapshot = store.snapshot(token)
                if snapshot is None:
                    break
                step = snapshot.get("step")
                marker = (step, snapshot.get("state"), snapshot.get("terminal_ranks"), snapshot.get("updated_at"))
                if marker != last_marker:
                    payload = json.dumps(_envelope(snapshot), allow_nan=False)
                    yield f"id: {step}\nevent: run\ndata: {payload}\n\n"
                    last_marker = marker
                else:
                    yield ": keepalive\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/events")
    async def global_events(request: Request):
        async def generate():
            previous = None
            while True:
                if await request.is_disconnected():
                    break
                rows = store.runs()
                compact = [
                    {
                        "token": row.get("token"),
                        "run_id": row.get("run_id"),
                        "state": row.get("state"),
                        "step": row.get("step"),
                        "updated_at": row.get("updated_at"),
                    }
                    for row in rows
                ]
                marker = json.dumps(compact, sort_keys=True)
                if marker != previous:
                    yield f"event: runs\ndata: {json.dumps(_envelope(compact), allow_nan=False)}\n\n"
                    previous = marker
                else:
                    yield ": keepalive\n\n"
                await asyncio.sleep(2.0)

        return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/compare")
    def compare(
        run_ids: str,
        keys: Optional[str] = None,
        x: str = "step",
        max_points: int = Query(2000, ge=10, le=50000),
    ):
        tokens = [item.strip() for item in run_ids.split(",") if item.strip()]
        if not tokens:
            raise HTTPException(status_code=400, detail="run_ids is required")
        result = {}
        for token in tokens:
            state = store.get(token)
            if state is None:
                continue
            data = store.series(
                token,
                _keys(keys),
                x_axis="time" if x == "time" else "step",
                max_points=max_points,
            )
            result[state.ref.token] = {
                "run_id": state.ref.display_name,
                "series": data,
            }
        return _envelope({"runs": result})

    @app.get("/api/export/{token}.csv")
    def export_csv(token: str, keys: Optional[str] = None):
        _require(store, token)
        data = store.series(token, _keys(keys), max_points=200000)
        if not data:
            raise HTTPException(status_code=404, detail="no series available")
        names = list(data)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["step"] + names)
        length = max((len(data[name]["x"]) for name in names), default=0)
        for index in range(length):
            # All series share the same aggregate steps; the first series supplies the x column.
            step = data[names[0]]["x"][index] if index < len(data[names[0]]["x"]) else ""
            values = [data[name]["y"][index] if index < len(data[name]["y"]) else "" for name in names]
            writer.writerow([step] + values)
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{token}.csv"'},
        )

    @app.get("/")
    def index():
        path = static_dir / "index.html"
        if not path.is_file():
            return JSONResponse({"detail": "dashboard assets are missing"}, status_code=500)
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    return app

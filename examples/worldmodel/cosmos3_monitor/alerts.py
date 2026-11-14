"""Dashboard-only alerting: turn monitor issues and training events into a deduplicated history.

There is deliberately no webhook or email here.  Every alert is visible in the dashboard, and the history is
persisted under the monitor's state directory so a restart does not forget what happened.  Run directories are
never written by the monitor.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .contracts import atomic_write_json

_SEVERITY_ORDER = {"critical": 0, "error": 1, "warn": 2, "info": 3}
_EVENT_ACTIVE_SECONDS = 600.0
_EVENT_ALERT_KINDS = {
    "nonfinite_loss",
    "loss_spike",
    "loss_drop",
    "step_time_jump",
    "rank_skew",
    "data_starvation",
    "memory_pressure",
}


class AlertEngine:
    """Evaluate issues/events for one run and keep active, resolved and acknowledged state."""

    def __init__(self, state_dir) -> None:
        self.state_dir = Path(state_dir)
        self.alerts_dir = self.state_dir / "alerts"
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self._active: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = {}
        self._acks = self._load_acks()

    # -- persistence --------------------------------------------------------------------------------------

    def _acks_path(self) -> Path:
        return self.state_dir / "acks.json"

    def _load_acks(self) -> Dict[str, float]:
        try:
            value = json.loads(self._acks_path().read_text(encoding="utf-8"))
            return {str(key): float(item) for key, item in value.items()} if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_acks(self) -> None:
        atomic_write_json(self._acks_path(), self._acks)

    def _history_path(self, token: str) -> Path:
        return self.alerts_dir / f"{token}.jsonl"

    def _append_history(self, token: str, alert: Dict[str, Any]) -> None:
        try:
            with self._history_path(token).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(alert, ensure_ascii=False, allow_nan=False) + "\n")
        except OSError:
            pass
        self._history.setdefault(token, []).append(alert)
        if len(self._history[token]) > 500:
            del self._history[token][: len(self._history[token]) - 500]

    def history(self, token: str, limit: int = 200) -> List[Dict[str, Any]]:
        rows = self._history.get(token)
        if rows is None:
            rows = []
            try:
                for line in self._history_path(token).read_text(encoding="utf-8").splitlines():
                    try:
                        value = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(value, dict):
                        rows.append(value)
            except OSError:
                pass
            if len(rows) > 500:
                rows = rows[-500:]
            self._history[token] = rows
        return list(reversed(rows[-max(1, min(limit, 500)) :]))

    # -- evaluation ---------------------------------------------------------------------------------------

    def evaluate(
        self,
        token: str,
        run_id: str,
        issues: Iterable[Dict[str, Any]],
        events: Iterable[Dict[str, Any]],
        now: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        now = time.time() if now is None else now
        active = self._active.setdefault(token, {})
        seen: set = set()

        def emit(
            *,
            rule: str,
            severity: str,
            message: str,
            subject: str = "",
            step: Optional[int] = None,
            host: Optional[str] = None,
            device: Optional[Any] = None,
            transient: bool = False,
        ) -> None:
            key = f"{rule}:{subject}"
            seen.add(key)
            entry = active.get(key)
            if entry is None:
                entry = {
                    "id": f"{token}-{rule}-{subject}".replace("/", "-"),
                    "rule": rule,
                    "run_id": run_id,
                    "severity": severity,
                    "subject": subject,
                    "message": message,
                    "step": step,
                    "host": host,
                    "device": device,
                    "first_seen": now,
                    "last_seen": now,
                    "count": 1,
                    "state": "active",
                    "acked": entry_acked(self._acks, f"{token}-{rule}-{subject}".replace("/", "-")),
                }
                entry["active_until"] = now + _EVENT_ACTIVE_SECONDS if transient else None
                active[key] = entry
                self._append_history(token, dict(entry))
            else:
                entry.update(
                    severity=severity,
                    message=message,
                    last_seen=now,
                    count=int(entry.get("count") or 0) + 1,
                )
                if step is not None:
                    entry["step"] = step

        for issue in issues:
            kind = str(issue.get("kind") or "issue")
            severity = str(issue.get("severity") or "warn")
            subject = str(issue.get("rank") if issue.get("rank") is not None else issue.get("host") or issue.get("step") or "")
            emit(
                rule=kind,
                severity=severity,
                message=str(issue.get("message") or kind),
                subject=subject,
                step=issue.get("step"),
                host=issue.get("host"),
                device=issue.get("device"),
                transient=False,
            )

        for event in events:
            severity = str(event.get("severity") or "warn")
            if severity not in ("warn", "error", "critical"):
                continue
            kind = str(event.get("kind") or "event")
            if kind not in _EVENT_ALERT_KINDS:
                continue
            try:
                event_time = float(event.get("time") or now)
            except (TypeError, ValueError):
                event_time = now
            # A completed run keeps its history; re-alerting on an event from an hour ago would make an old
            # failure look like a current one.  Only freshly written events become transient active alerts.
            if now - event_time > _EVENT_ACTIVE_SECONDS:
                continue
            step = event.get("step")
            rank = event.get("rank")
            subject = f"{step}:{rank}" if rank is not None else str(step)
            emit(
                rule=kind,
                severity=severity,
                message=str(event.get("message") or kind),
                subject=subject,
                step=step,
                transient=True,
            )

        for key, entry in list(active.items()):
            transient = entry.get("active_until") is not None
            expired = transient and now > float(entry["active_until"])
            if key not in seen and (not transient or expired):
                entry["state"] = "resolved"
                entry["resolved_at"] = now
                self._append_history(token, dict(entry))
                del active[key]

        return sorted(
            active.values(),
            key=lambda item: (_SEVERITY_ORDER.get(str(item.get("severity")), 9), -float(item.get("last_seen") or 0)),
        )

    # -- acknowledgement ----------------------------------------------------------------------------------

    def ack(self, token: str, alert_id: str, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        for entry in self._active.get(token, {}).values():
            if entry.get("id") == alert_id:
                entry["acked"] = True
                self._acks[alert_id] = now
                self._save_acks()
                return True
        # An already-resolved alert can still be acknowledged in the history view.
        self._acks[alert_id] = now
        self._save_acks()
        return True

    def view(self, token: str) -> Dict[str, Any]:
        active = self._active.get(token, {})
        return {
            "active": sorted(
                active.values(),
                key=lambda item: (_SEVERITY_ORDER.get(str(item.get("severity")), 9), -float(item.get("last_seen") or 0)),
            ),
            "history": self.history(token, limit=200),
            "counts": {
                severity: sum(1 for item in active.values() if item.get("severity") == severity)
                for severity in ("critical", "error", "warn", "info")
            },
        }


def entry_acked(acks: Dict[str, float], alert_id: str) -> bool:
    return alert_id in acks

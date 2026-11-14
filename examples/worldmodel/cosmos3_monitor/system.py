"""Node-level telemetry: parse ``npu-smi`` and ``/proc``, sample the host, keep a bounded history.

AICore utilization is deliberately not parsed.  The 25.2.1 driver on this cluster reports 89-99% even when
only dead processes remain (see ``colossalai/cluster/node_guard.py``), so a chart or alert built on it would
be confidently wrong.  HBM usage, temperature, power and the process table are the reliable signals.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .readers import file_age

_DEVICE_HEADER = re.compile(
    r"^\|\s*(\d+)\s+([A-Za-z0-9_+-]+)\s*\|\s*([A-Za-z]+)\s*\|\s*([\d.]+)\s+(\d+)\s+(\d+)\s*/\s*(\d+)\s*\|"
)
_DEVICE_LINE = re.compile(
    r"^\|\s*(\d+)\s*\|\s*([0-9A-Fa-f:.]+)\s*\|\s*([\d.]+)\s+(\d+)\s*/\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s*\|"
)
_PROCESS_LINE = re.compile(r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*([^|]*)\|\s*(\d+)\s*\|")
_NO_PROCESS = re.compile(r"No running processes found in NPU\s+(\d+)", re.IGNORECASE)


def parse_npu_smi(text: str) -> Dict[str, Any]:
    """Parse ``npu-smi info`` into devices with HBM, temperature, power and processes.

    The parser is line-oriented rather than table-oriented so a driver that adds a column or changes spacing
    degrades to one missing field instead of an exception.  Unknown lines are ignored.
    """
    devices: Dict[int, Dict[str, Any]] = {}
    errors: List[str] = []
    current_index: Optional[int] = None
    for line in text.splitlines():
        header = _DEVICE_HEADER.match(line.strip())
        if header:
            index = int(header.group(1))
            current_index = index
            device = devices.setdefault(index, {"index": index, "processes": []})
            device.update(
                name=header.group(2),
                health=header.group(3),
                power_w=_float(header.group(4)),
                temperature_c=_float(header.group(5)),
                hbm_used_mib=float(header.group(7)),
                hbm_total_mib=float(header.group(7)),
            )
            continue
        detail = _DEVICE_LINE.match(line.strip())
        if detail:
            # The first column of the detail row is the *chip* index (always 0 on a single-die 910B2); the NPU
            # index is the one from the preceding header row.
            index = current_index if current_index is not None else int(detail.group(1))
            device = devices.setdefault(index, {"index": index, "processes": []})
            device.update(
                bus_id=detail.group(2),
                hbm_used_mib=_float(detail.group(6)),
                hbm_total_mib=_float(detail.group(7)),
            )
            continue
        process = _PROCESS_LINE.match(line.strip())
        if process:
            npu_index = int(process.group(1))
            device = devices.setdefault(npu_index, {"index": npu_index, "processes": []})
            device["processes"].append(
                {
                    "pid": int(process.group(3)),
                    "name": process.group(4).strip() or None,
                    "used_mib": _float(process.group(5)),
                }
            )
            continue
        missing = _NO_PROCESS.search(line)
        if missing:
            index = int(missing.group(1))
            devices.setdefault(index, {"index": index, "processes": []})
    if not devices:
        errors.append("npu-smi output contained no device rows")
    return {"devices": [devices[index] for index in sorted(devices)], "errors": errors}


def _float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def npu_smi_text(timeout: float = 5.0) -> Tuple[Optional[str], Optional[str]]:
    try:
        result = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError:
        return None, "npu-smi not found"
    except subprocess.TimeoutExpired:
        return None, f"npu-smi timed out after {timeout:.0f}s"
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if result.returncode != 0:
        return None, f"npu-smi exited {result.returncode}: {(result.stderr or '').strip()[:200]}"
    return result.stdout, None


class HostSampler:
    """Sample CPU/memory/network/disk and the NPU table, keeping the previous /proc counters.

    The sampler lives in the collector process, never in training.  ``npu-smi`` can be slow on a busy node,
    so a failed or slow call degrades that part of the sample without dropping the host counters.
    """

    def __init__(self, disk_paths: Optional[List[str]] = None, entry: str = "train_sft.py", run_out: str = "") -> None:
        self.disk_paths = [path for path in (disk_paths or []) if path]
        self.entry = entry
        self.run_out = run_out
        self._cpu_prev: Optional[Tuple[int, int]] = None
        self._net_prev: Optional[Tuple[int, int, float]] = None
        self._last_npu_text: Optional[str] = None
        self._last_npu_error: Optional[str] = None
        self._last_npu_at = 0.0
        self.npu_interval_s = 5.0
        self._npu_slow = False

    # -- host ---------------------------------------------------------------------------------------------

    def _cpu_percent(self) -> Optional[float]:
        try:
            fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
            values = [int(item) for item in fields]
        except (OSError, ValueError, IndexError):
            return None
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values)
        previous = self._cpu_prev
        self._cpu_prev = (idle, total)
        if previous is None:
            return None
        idle_delta = idle - previous[0]
        total_delta = total - previous[1]
        if total_delta <= 0:
            return None
        return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))

    def _memory(self) -> Dict[str, Optional[float]]:
        values: Dict[str, Optional[float]] = {}
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    values[key] = float(rest.strip().split()[0]) / 1024 / 1024  # KiB -> GiB
        except (OSError, ValueError):
            pass
        return {
            "total_gib": values.get("MemTotal"),
            "available_gib": values.get("MemAvailable"),
            "used_gib": (
                values["MemTotal"] - values["MemAvailable"]
                if "MemTotal" in values and "MemAvailable" in values
                else None
            ),
        }

    def _network(self) -> Optional[Dict[str, Any]]:
        interface = os.environ.get("HCCL_SOCKET_IFNAME") or "bond1"
        try:
            for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
                name, _, rest = line.partition(":")
                if name.strip() != interface:
                    continue
                fields = rest.split()
                rx, tx = int(fields[0]), int(fields[8])
                now = time.time()
                previous = self._net_prev
                self._net_prev = (rx, tx, now)
                rate = None
                if previous is not None and now > previous[2]:
                    elapsed = now - previous[2]
                    rate = {
                        "rx_bytes_per_s": max(0.0, (rx - previous[0]) / elapsed),
                        "tx_bytes_per_s": max(0.0, (tx - previous[1]) / elapsed),
                    }
                return {"interface": interface, "rx_bytes": rx, "tx_bytes": tx, "rate": rate}
        except (OSError, ValueError, IndexError):
            return None
        return None

    def _disks(self) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for raw in self.disk_paths:
            path = Path(raw)
            probe = path
            while not probe.exists() and probe.parent != probe:
                probe = probe.parent
            try:
                usage = shutil.disk_usage(str(probe))
            except OSError:
                continue
            result[str(path)] = {
                "free_gib": usage.free / 2**30,
                "total_gib": usage.total / 2**30,
            }
        return result

    def _shm(self) -> Dict[str, Optional[float]]:
        try:
            usage = shutil.disk_usage("/dev/shm")
        except OSError:
            return {"free_gib": None, "total_gib": None}
        return {"free_gib": usage.free / 2**30, "total_gib": usage.total / 2**30}

    # -- processes ----------------------------------------------------------------------------------------

    def _training_processes(self) -> List[Dict[str, Any]]:
        """Find this run's training processes, and map them to ranks when the environment says so."""
        rows: List[Dict[str, Any]] = []
        for proc in Path("/proc").glob("[0-9]*"):
            try:
                argv = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
            except OSError:
                continue
            argv = [item for item in argv if item]
            if not argv:
                continue
            joined = " ".join(argv)
            if self.entry and self.entry not in joined:
                continue
            if self.run_out and self.run_out not in joined:
                continue
            rank = local_rank = None
            try:
                for entry in (proc / "environ").read_bytes().split(b"\0"):
                    if entry.startswith(b"RANK="):
                        rank = int(entry[5:] or -1)
                    elif entry.startswith(b"LOCAL_RANK="):
                        local_rank = int(entry[11:] or -1)
            except (OSError, ValueError):
                pass
            row: Dict[str, Any] = {
                "pid": int(proc.name),
                "rank": rank,
                "local_rank": local_rank,
                "command": joined[:300],
            }
            try:
                stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
                row["state"] = stat[0]
                try:
                    row["wchan"] = (proc / "wchan").read_text().strip() or None
                except OSError:
                    row["wchan"] = None
            except (OSError, IndexError):
                pass
            rows.append(row)
        rows.sort(key=lambda item: (item.get("rank") is None, item.get("rank") or 0, item["pid"]))
        return rows

    # -- NPU ----------------------------------------------------------------------------------------------

    def _npus(self) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        now = time.time()
        if now - self._last_npu_at < self.npu_interval_s and self._last_npu_text is not None:
            return parse_npu_smi(self._last_npu_text)["devices"], self._last_npu_error
        started = time.perf_counter()
        text, error = npu_smi_text()
        elapsed = time.perf_counter() - started
        # A slow npu-smi must not dominate a 5 s sample.  Back off to 15 s when it takes more than 2 s.
        self._npu_slow = elapsed > 2.0
        self.npu_interval_s = 15.0 if self._npu_slow else 5.0
        self._last_npu_at = now
        self._last_npu_error = error
        if text is None:
            return [], error
        self._last_npu_text = text
        return parse_npu_smi(text)["devices"], None

    def sample(self, node_rank: int = 0) -> Dict[str, Any]:
        devices, npu_error = self._npus()
        processes = self._training_processes()
        by_rank = {row["rank"]: row for row in processes if row.get("rank") is not None}
        for device in devices:
            for process in device.get("processes", []):
                match = next((row for row in processes if row["pid"] == process["pid"]), None)
                if match is not None:
                    process["rank"] = match.get("rank")
                    process["local_rank"] = match.get("local_rank")
                    process["state"] = match.get("state")
                    process["wchan"] = match.get("wchan")
        errors = [npu_error] if npu_error else []
        try:
            load1, load5, _ = os.getloadavg()
        except (OSError, AttributeError):
            load1 = load5 = None
        payload = {
            "type": "system",
            "schema": 1,
            "time": time.time(),
            "hostname": socket.gethostname(),
            "node_rank": int(node_rank),
            "devices": devices,
            "host": {
                "cpu_percent": self._cpu_percent(),
                "load1": load1,
                "load5": load5,
                "memory": self._memory(),
                "disk_free_gib": self._disks(),
                "shm": self._shm(),
                "net": self._network(),
            },
            "training_processes": len(processes),
            "processes": processes,
            "ranked_processes": [by_rank[rank] for rank in sorted(by_rank)],
            "collector": {
                "pid": os.getpid(),
                "interval_s": None,
                "npu_interval_s": self.npu_interval_s,
                "errors": errors,
            },
            "advisory": {"aicore_utilization_excluded": True},
        }
        return payload


class SystemHistory:
    """Bounded per-host history for the System tab and the system alert rules."""

    def __init__(self, max_points: int = 2000) -> None:
        self.max_points = int(max_points)
        self.hosts: Dict[str, Dict[str, Any]] = {}

    def ingest(self, record: Dict[str, Any]) -> None:
        host = str(record.get("hostname") or "unknown")
        entry = self.hosts.setdefault(host, {"latest": None, "points": [], "devices": {}})
        entry["latest"] = record
        entry["points"].append(record)
        if len(entry["points"]) > self.max_points:
            del entry["points"][: len(entry["points"]) - self.max_points]
        for device in record.get("devices") or []:
            index = str(device.get("index"))
            series = entry["devices"].setdefault(index, {"hbm_used_mib": [], "temperature_c": [], "power_w": []})
            for key in ("hbm_used_mib", "temperature_c", "power_w"):
                value = device.get(key)
                if isinstance(value, (int, float)):
                    values = series[key]
                    values.append([record.get("time"), float(value)])
                    if len(values) > self.max_points:
                        del values[: len(values) - self.max_points]

    def latest(self, host: Optional[str] = None) -> List[Dict[str, Any]]:
        if host:
            entry = self.hosts.get(host)
            return [entry["latest"]] if entry and entry["latest"] else []
        return [entry["latest"] for entry in self.hosts.values() if entry["latest"]]

    def snapshot(self, host: Optional[str] = None) -> Dict[str, Any]:
        hosts = {}
        for name, entry in self.hosts.items():
            if host and name != host:
                continue
            hosts[name] = {
                "hostname": name,
                "latest": entry["latest"],
                "age_s": file_age_latest(entry["latest"]),
                "devices": {
                    index: {key: values[-500:] for key, values in series.items()}
                    for index, series in entry["devices"].items()
                },
            }
        return {"hosts": hosts}


def file_age_latest(record: Optional[Dict[str, Any]]) -> Optional[float]:
    if not record:
        return None
    timestamp = record.get("time")
    try:
        return max(0.0, time.time() - float(timestamp))
    except (TypeError, ValueError):
        return None


def read_system_files(metrics_dir, history: SystemHistory) -> None:
    """Tail every ``system/node-*.jsonl`` under a metrics directory into ``history``."""
    directory = Path(metrics_dir) / "system"
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("node-*.jsonl")):
        offset = getattr(history, "_offsets", {}).get(str(path), 0)
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size < offset:
            offset = 0
        try:
            with path.open("rb") as stream:
                stream.seek(offset)
                chunk = stream.read(size - offset)
        except OSError:
            continue
        if not hasattr(history, "_offsets"):
            history._offsets = {}
        history._offsets[str(path)] = size
        for line in chunk.split(b"\n"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                history.ingest(record)

#!/usr/bin/env python3
"""Select idle GPUs and run one ColossalAI CI batch while holding a CI lock."""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

GPU_UUID = re.compile(r"GPU-[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}\Z")
GPU_COUNTS = {
    "2": 1,
    "3": 1,
    "4a": 1,
    "4b": 1,
    "4c": 1,
    "4d": 1,
    "4e": 1,
    "5": 2,
    "6": 4,
    "7": 4,
    "8": 4,
    "9": 4,
    "10": 8,
    "11": 2,
}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def parse_pool(value):
    if not value:
        return None
    items = value.split(",")
    if any(not item.isdigit() for item in items) or len(items) != len(set(items)):
        raise RuntimeError("COLOSSAL_GPU_POOL must be a comma-separated list of unique GPU indices")
    return {int(item) for item in items}


def find_idle(inventory, processes, allowed=None):
    busy = set()
    for line in processes.splitlines():
        value = line.strip()
        if value:
            if not GPU_UUID.fullmatch(value):
                raise RuntimeError("GPU process information is unknown")
            busy.add(value)

    candidates = []
    seen = set()
    for line in inventory.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError("Unexpected GPU inventory format")
        index, gpu, memory, utilization = fields
        if not GPU_UUID.fullmatch(gpu) or gpu in seen:
            raise RuntimeError("Invalid or duplicate GPU UUID")
        seen.add(gpu)
        if not all(value.isdigit() for value in (index, memory, utilization)):
            raise RuntimeError("GPU occupancy is unknown")
        index = int(index)
        if allowed is not None and index not in allowed:
            continue
        if gpu not in busy and int(memory) <= 256 and int(utilization) == 0:
            candidates.append((index, gpu))

    candidates.sort()
    return candidates


def select_idle(inventory, processes, count, allowed=None):
    candidates = find_idle(inventory, processes, allowed)
    if len(candidates) < count:
        raise RuntimeError(f"Need {count} idle GPU(s), found {len(candidates)} in the authorized pool")
    return candidates[:count]


def query_gpus():
    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
        timeout=15,
    ).stdout
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
        timeout=15,
    ).stdout
    return inventory, processes


def stream_process(command, log_path, env):
    child = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        start_new_session=True,
    )

    def interrupted(signum, frame):
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise RuntimeError(f"Interrupted by signal {signum}")

    old_term = signal.signal(signal.SIGTERM, interrupted)
    old_int = signal.signal(signal.SIGINT, interrupted)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            for line in child.stdout:
                sys.stdout.write(line)
                log.write(line)
        return child.wait()
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, choices=sorted(GPU_COUNTS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path("/tmp/colossalai-ci-gpu.lock"))
    args = parser.parse_args()

    if os.name != "posix":
        raise RuntimeError("GPU batches require a Linux runner")
    import fcntl

    args.output.mkdir(parents=True, exist_ok=True)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "batch": args.batch,
        "status": "failed",
        "host": socket.gethostname().split(".")[0],
        "runner": os.environ.get("RUNNER_NAME"),
        "source_sha": os.environ.get("CI_SOURCE_SHA", os.environ.get("GITHUB_SHA")),
        "started_at": utcnow(),
    }

    try:
        with args.lock.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            inventory, processes = query_gpus()
            selected = select_idle(
                inventory,
                processes,
                GPU_COUNTS[args.batch],
                parse_pool(os.environ.get("COLOSSAL_GPU_POOL")),
            )
            result["gpu_indices"] = [index for index, _ in selected]
            result["gpu_uuids"] = [gpu for _, gpu in selected]
            print("Selected idle GPUs: " + json.dumps(selected), flush=True)

            # Check once more while the CI lock is held, immediately before launch.
            inventory, processes = query_gpus()
            still_idle = select_idle(
                inventory,
                processes,
                GPU_COUNTS[args.batch],
                {index for index, _ in selected},
            )
            if [index for index, _ in still_idle] != result["gpu_indices"]:
                raise RuntimeError("GPU occupancy changed before the batch could start")

            script = Path(__file__).with_name("run_colossalai_batch.sh")
            gpu_list = ",".join(str(index) for index, _ in selected)
            env = os.environ.copy()
            code = stream_process(["bash", str(script), args.batch, gpu_list], args.output / "pytest.log", env)
            if code != 0:
                raise RuntimeError(f"Batch {args.batch} failed with exit code {code}")
            result["status"] = "passed"
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
    finally:
        result["finished_at"] = utcnow()
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

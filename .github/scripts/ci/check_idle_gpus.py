#!/usr/bin/env python3
"""Report idle H20 capacity for the hourly layered-CI gate."""

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_gpu_batch import find_idle, parse_pool, query_gpus


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def readiness(idle_count, layer2_required=2, layer3_required=8):
    if idle_count < 0 or layer2_required < 1 or layer3_required < layer2_required:
        raise ValueError("Invalid GPU readiness thresholds")
    return {
        "layer2_ready": idle_count >= layer2_required,
        "layer3_ready": idle_count >= layer3_required,
    }


def append_outputs(path, values):
    if path is None:
        return
    with path.open("a", encoding="utf-8") as output:
        for key, value in values.items():
            if isinstance(value, bool):
                value = str(value).lower()
            output.write(f"{key}={value}\n")


def append_summary(path, result):
    if path is None:
        return
    with path.open("a", encoding="utf-8") as summary:
        summary.write("## Hourly GPU capacity gate\n\n")
        summary.write(f"- Host: `{result['host']}`\n")
        summary.write(f"- Runner: `{result['runner']}`\n")
        summary.write(f"- Idle GPU indices: `{result['idle_gpu_indices']}`\n")
        summary.write(f"- Layer 2 ready (needs 2): `{str(result['layer2_ready']).lower()}`\n")
        summary.write(f"- Layer 3 ready (needs 8): `{str(result['layer3_ready']).lower()}`\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer2-required", type=int, default=2)
    parser.add_argument("--layer3-required", type=int, default=8)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(os.environ.get("CI_GPU_LOCK", "/tmp/colossalai-ci-gpu.lock")),
    )
    parser.add_argument("--github-output", type=Path, default=os.environ.get("GITHUB_OUTPUT"))
    parser.add_argument("--summary", type=Path, default=os.environ.get("GITHUB_STEP_SUMMARY"))
    args = parser.parse_args()

    if os.name != "posix":
        raise RuntimeError("GPU capacity checks require a Linux runner")
    import fcntl

    args.lock.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "host": socket.gethostname().split(".")[0],
        "runner": os.environ.get("RUNNER_NAME"),
        "checked_at": utcnow(),
    }

    try:
        with args.lock.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            inventory, processes = query_gpus()
            idle = find_idle(inventory, processes, parse_pool(os.environ.get("COLOSSAL_GPU_POOL")))
        result["idle_gpu_count"] = len(idle)
        result["idle_gpu_indices"] = [index for index, _ in idle]
        result.update(readiness(len(idle), args.layer2_required, args.layer3_required))
        append_outputs(
            args.github_output,
            {
                "idle_gpu_count": result["idle_gpu_count"],
                "layer2_ready": result["layer2_ready"],
                "layer3_ready": result["layer3_ready"],
            },
        )
        append_summary(args.summary, result)
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
        print(json.dumps(result, indent=2), flush=True)
        return 1

    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

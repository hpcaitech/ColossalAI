#!/usr/bin/env python3
"""Immediate PR qualification using two currently idle GPUs on gpu-h20-5.

Temporary rollout for one trusted internal PR branch, under manual resource
authorization. This does not contact or claim a reservation from the website.
"""

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


def validate_event(event):
    pr = event["pull_request"]
    if (
        event["repository"]["full_name"] != "hpcaitech/ColossalAI"
        or pr["head"]["repo"]["full_name"] != "hpcaitech/ColossalAI"
        or pr["head"]["ref"] != "ci/e1-runner-bootstrap"
        or pr["base"]["ref"] != "main"
        or pr["user"]["login"] != "richardoo-707"
    ):
        raise RuntimeError("This qualification is restricted to the authorized internal E1 PR")
    return pr


def select_idle(inventory, processes):
    busy = set()
    for line in processes.splitlines():
        value = line.strip()
        if value:
            if not GPU_UUID.fullmatch(value):
                raise RuntimeError("GPU process information is unknown")
            busy.add(value)
    candidates, seen = [], set()
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
        if gpu not in busy and int(memory) <= 256 and int(utilization) == 0:
            candidates.append((int(index), gpu))
    if len(candidates) < 2:
        raise RuntimeError("Fewer than two idle GPUs; immediate test cannot run")
    candidates.sort()
    return candidates[:2]


def query_gpus():
    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
        timeout=10,
    ).stdout
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
        timeout=10,
    ).stdout
    return select_idle(inventory, processes)


def has_gpu_pass(log):
    for line in log.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and (
            record.get("result") == "E1_GPU_SMOKE_PASS"
            and record.get("gpu_tested") is True
            and record.get("backend") == "nccl"
            and record.get("world_size") == 2
        ):
            return True
    return False


def has_colossalai_pass(log):
    for line in log.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("result") == "E1_COLOSSALAI_PASS":
            if record.get("gpu_tested") is True and record.get("tests_passed") == 2:
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("smoke", "colossalai"), default="smoke")
    args = parser.parse_args()
    if os.name != "posix" or socket.gethostname().split(".")[0] != "gpu-h20-5":
        raise RuntimeError("This job must execute on gpu-h20-5")
    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request":
        raise RuntimeError("This entry point requires a GitHub pull_request event")
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    pr = validate_event(event)
    args.output.mkdir(parents=True, mode=0o700)
    result = {
        "status": "failed",
        "suite": args.suite,
        "gpu_tested": False,
        "website_contacted": False,
        "allocation_source": "manual_authorization_idle_selection",
        "host": "gpu-h20-5",
        "pull_request": pr["number"],
        "head_sha": pr["head"]["sha"],
        "checkout_sha": os.environ.get("GITHUB_SHA"),
        "source_method": "github_codeload_merge_commit",
        "source_archive_sha256": os.environ.get("E1_SOURCE_ARCHIVE_SHA256"),
        "runner": os.environ.get("RUNNER_NAME"),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    child = None

    def interrupted(signum, frame):
        raise RuntimeError(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        selected = query_gpus()
        result["gpu_indices"] = [index for index, gpu in selected]
        result["gpu_uuids"] = [gpu for index, gpu in selected]
        print("Selected idle GPUs: " + json.dumps(selected), flush=True)
        command = ["bash", str(Path(__file__).with_name("run_gpu_smoke.sh")), "--run", *result["gpu_uuids"]]
        if args.suite == "colossalai":
            command = [
                os.environ["E1_COLOSSALAI_PYTHON"],
                "-B",
                str(Path(__file__).with_name("colossalai_suite.py")),
                "--output",
                str(args.output / "colossalai"),
                "--gpus",
                *result["gpu_uuids"],
            ]
        env = {key: value for key, value in os.environ.items() if not key.startswith("E1_BOOKING_")}
        with (args.output / "test.log").open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            code = child.wait(timeout=360)
        log_text = (args.output / "test.log").read_text()
        print(log_text, flush=True)
        passed = has_gpu_pass(log_text) if args.suite == "smoke" else has_colossalai_pass(log_text)
        if code != 0 or not passed:
            raise RuntimeError(f"GPU test failed (exit={code}); no successful two-GPU qualification")
        result.update(status="passed", gpu_tested=True)
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)

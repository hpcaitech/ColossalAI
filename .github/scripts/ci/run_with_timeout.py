#!/usr/bin/env python3
"""Bound a test command and reap its worker group before a log tee waits for EOF."""

import argparse
import signal
import subprocess
import sys

from run_gpu_batch import terminate_process_group


def run_with_timeout(command, timeout_seconds):
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    child = subprocess.Popen(command, start_new_session=True)

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    old_term = signal.signal(signal.SIGTERM, interrupted)
    old_int = signal.signal(signal.SIGINT, interrupted)
    try:
        try:
            code = child.wait(timeout=timeout_seconds)
            return code if code >= 0 else 128 - code
        except subprocess.TimeoutExpired:
            print(f"Test command exceeded {timeout_seconds:g} seconds; cleaning its worker group", flush=True)
            return 124
    finally:
        # Clean grandchildren even when the test parent has already exited.
        try:
            terminate_process_group(child)
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.timeout_seconds <= 0:
        parser.error("a command and a positive timeout are required")
    return run_with_timeout(command, args.timeout_seconds)


if __name__ == "__main__":
    sys.exit(main())

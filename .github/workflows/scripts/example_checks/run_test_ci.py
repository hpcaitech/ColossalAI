#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


def _emit_error(title: str, message: str) -> None:
    print(f"::error title={title}::{message}")


def run_test_ci(script_path: Path) -> int:
    script_path = script_path.resolve()

    if not script_path.is_file():
        _emit_error("Missing example CI script", f"{script_path} does not exist")
        return 1

    if script_path.stat().st_size == 0:
        _emit_error(
            "Empty example CI script",
            f"{script_path} is empty; an empty Bash script exits successfully without running a test",
        )
        return 1

    print(f"Running {script_path}", flush=True)
    completed = subprocess.run(["bash", script_path.name], cwd=script_path.parent, check=False)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Reject missing or empty example CI scripts before running them")
    parser.add_argument("script", type=Path, help="Path to an example test_ci.sh")
    args = parser.parse_args()
    return run_test_ci(args.script)


if __name__ == "__main__":
    raise SystemExit(main())

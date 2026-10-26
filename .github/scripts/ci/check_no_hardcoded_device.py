#!/usr/bin/env python3
"""Fail if guarded directories hardcode a specific accelerator or communication backend.

Device-specific branches belong in the accelerator, communication backend and extension
layers only. Elsewhere they fail silently: the code works on CUDA and breaks the first time
it runs anywhere else. For example, TorchFSDPPlugin.configure() used to pass
``torch.cuda.current_device()`` as the FSDP ``device_id``, so ``booster.boost()`` raised
``AssertionError: Torch not compiled with CUDA enabled`` on an Ascend NPU.

Directories are locked down one at a time: clean one up, add it to GUARDED, and this keeps
it clean. Much of the tree still hardcodes CUDA (mostly under ``colossalai/legacy``) and is
deliberately out of scope.

Usage:
    python .github/scripts/ci/check_no_hardcoded_device.py            # check GUARDED
    python .github/scripts/ci/check_no_hardcoded_device.py <dir>...   # check given dirs

To keep a genuinely device-specific line, mark it with a reason:
    some_call(torch.cuda.current_device())  # allow-hardcoded-device: CUDA-only FP8 path
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Directories that have been cleaned up and must stay device-agnostic, relative to the repo root.
GUARDED = [
    "colossalai/booster/plugin",
]

# (pattern, what was found, what to use instead)
PATTERNS = [
    (re.compile(r"\btorch\.cuda\."), "torch.cuda.*", "the matching get_accelerator() method"),
    (re.compile(r"\.cuda\(\)"), ".cuda()", ".to(get_accelerator().get_current_device())"),
    (re.compile(r"""(?<![\w.])["']nccl["']"""), "hardcoded nccl backend", "the backend the accelerator reports"),
]

ALLOW = re.compile(r"#\s*allow-hardcoded-device:")
REPO_ROOT = Path(__file__).resolve().parents[3]


def scan(directory: Path) -> list[tuple[Path, int, str, str, str]]:
    """Return (path, lineno, source line, what was found, suggested fix) for every violation."""
    violations = []
    for path in sorted(directory.rglob("*.py")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            # Report instead of skipping silently: an unreadable file is not a passing file.
            print(f"error: cannot read {path}: {exc}", file=sys.stderr)
            raise SystemExit(2)
        for lineno, line in enumerate(lines, 1):
            if ALLOW.search(line):
                continue
            for pattern, found, fix in PATTERNS:
                if pattern.search(line):
                    violations.append((path, lineno, line.strip(), found, fix))
    return violations


def main() -> int:
    targets = sys.argv[1:] or GUARDED
    missing = [t for t in targets if not (REPO_ROOT / t).is_dir()]
    if missing:
        print(f"error: no such directory: {', '.join(missing)} (repo root {REPO_ROOT})", file=sys.stderr)
        return 2

    violations = [v for target in targets for v in scan(REPO_ROOT / target)]
    if not violations:
        print(f"no hardcoded device found in {len(targets)} guarded director{'y' if len(targets) == 1 else 'ies'}")
        return 0

    print(f"Found {len(violations)} hardcoded device reference(s) in guarded directories:\n", file=sys.stderr)
    for path, lineno, line, found, fix in violations:
        print(f"  {path.relative_to(REPO_ROOT)}:{lineno}: {found}", file=sys.stderr)
        print(f"      {line}", file=sys.stderr)
        print(f"      use {fix} instead\n", file=sys.stderr)
    print(
        "Device-specific branches belong in the accelerator, communication backend and\n"
        "extension layers only. To keep a line that is genuinely device-specific, mark it\n"
        "with '# allow-hardcoded-device: <reason>'.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

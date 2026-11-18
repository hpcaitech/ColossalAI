#!/usr/bin/env python3
"""Record or verify the exact model, plugin, numerical-tool and test sources."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCES = (
    "colossalai/shardformer/modeling/ouro.py",
    "colossalai/shardformer/policies/_remote.py",
    "colossalai/shardformer/policies/ouro.py",
    "colossalai/shardformer/policies/auto_policy.py",
    "colossalai/booster",
    "colossalai/checkpoint_io",
    "colossalai/zero",
    "tests/kit/numerics",
    "tests/kit/remote_models.py",
    "tests/kit/ouro.py",
    "tests/kit/model_zoo/transformers/ouro.py",
    "tests/fixtures/ouro",
    "tests/test_model_zoo/test_ouro.py",
    "tests/test_booster/test_plugin/test_ouro.py",
    "tests/test_checkpoint_io/test_gemini_offload_export.py",
    "tests/test_checkpoint_io/test_gemini_master_resume.py",
    "scripts/phase2/2d4/run_tests.sh",
    "scripts/phase2/2d4/run_host.sh",
    "scripts/phase2/2d4/run.sh",
    "scripts/phase2/2d4/run_acceptance.sh",
    "scripts/phase2/2d4/check_full_ouro.py",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path)
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        manifest = json.loads(args.verify.read_text())
        failed = []
        for name, expected in manifest["sha256"].items():
            path = ROOT / name
            if not path.is_file() or digest(path) != expected:
                failed.append(name)
        print(json.dumps(dict(passed=not failed, checked=len(manifest["sha256"]), mismatches=failed)))
        return int(bool(failed))
    paths = set()
    for name in SOURCES:
        path = ROOT / name
        paths.update(path.rglob("*") if path.is_dir() else [path])
    hashes = {
        str(p.relative_to(ROOT)): digest(p)
        for p in sorted(paths)
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    result = dict(
        base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        sha256=hashes,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(files=len(hashes), manifest=str(args.out))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

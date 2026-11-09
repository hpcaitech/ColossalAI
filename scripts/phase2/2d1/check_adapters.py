"""Run the production adapter regressions in the development container."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import torch
import transformers

ROOT = Path(__file__).resolve().parents[3]


def main():
    print("Torch:", torch.__version__, "Transformers:", transformers.__version__, flush=True)
    if transformers.__version__ != "5.17.0":
        raise RuntimeError("These adapter regressions require Transformers 5.17.0; refusing a skipped run")
    print("Scope: CPU / FP32 / tiny / eager / no KV cache / native ShardFormer", flush=True)
    paths = [
        "colossalai/shardformer/modeling/ouro.py",
        "colossalai/shardformer/modeling/llada.py",
        "colossalai/shardformer/modeling/llada2_moe.py",
        "colossalai/shardformer/policies/auto_policy.py",
        "colossalai/shardformer/policies/_remote.py",
        "colossalai/shardformer/policies/llada.py",
        "colossalai/shardformer/policies/llada2_moe.py",
        "colossalai/shardformer/policies/ouro.py",
        "tests/kit/remote_models.py",
        "tests/test_shardformer/test_model/test_shard_remote_models.py",
    ]
    for name in paths:
        print(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), name, flush=True)
    env = dict(os.environ, COLOSSALAI_REQUIRE_REMOTE_MODELS="1")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "pytest",
            "-v",
            "--tb=short",
            "tests/test_shardformer/test_model/test_shard_remote_models.py",
            "tests/test_shardformer/test_hf5_compat/test_d4_compat.py",
            *sys.argv[1:],
        ],
        cwd=ROOT,
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())

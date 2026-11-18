#!/usr/bin/env bash
# Container entrypoint: persistent logs/JUnit and the actual pytest exit status.
set -euo pipefail
cd /workspace/ColossalAI
out="$(mktemp -d "/results/2d4/pytest-$(date -u +%Y%m%d-%H%M%S)-XXXXXX")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 COLOSSALAI_REQUIRE_REMOTE_MODELS=1
if [[ "${D2D4_OFFLINE_FIXTURES:-0}" == 1 ]]; then
    export HF_HUB_CACHE="$out/empty-hub" HF_MODULES_CACHE="$out/modules"
    mkdir -p "$HF_HUB_CACHE" "$HF_MODULES_CACHE"
fi
printf 'Results: %s\n' "$out"
{
    git rev-parse HEAD
    git diff --stat
    sha256sum colossalai/shardformer/modeling/ouro.py colossalai/shardformer/policies/{ouro,_remote}.py tests/kit/ouro.py
    timeout --signal=TERM --kill-after=30s "${D2D4_TEST_TIMEOUT:-1800}s" python -m pytest "$@" -ra --basetemp="$out/tmp" --junitxml="$out/junit.xml"
} 2>&1 | tee "$out/run.log"

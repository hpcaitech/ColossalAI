#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ColossalAI
out="$(mktemp -d /results/2d4/ci-checks-XXXXXX)"
printf 'Results: %s\n' "$out"
{
    bash -n .github/scripts/ci/run_colossalai_batch.sh
    bash -n scripts/phase2/2d4/run{,_host,_tests,_acceptance}.sh
    python -B -m unittest discover -s .github/scripts/ci -p 'test_*.py'
} 2>&1 | tee "$out/run.log"

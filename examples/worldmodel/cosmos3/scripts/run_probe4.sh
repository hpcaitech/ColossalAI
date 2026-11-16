#!/bin/bash
# After the memory check: measure how many bf16 outputs move if the DiT norm's row sum is replaced.
set -uo pipefail

LOCAL=/mnt/disk0/jiangzs/nzf
BENCH=$LOCAL/bench_online
REPO=$LOCAL/ColossalAI
PY=$LOCAL/cai-npu/bin/python

export PYTHONPATH=$REPO
cd "$REPO/examples/worldmodel/cosmos3" || exit 1

for _ in $(seq 1 240); do
  if grep -qa 'memcheck exit' "$BENCH/logs/memcheck_runner.log" 2>/dev/null; then break; fi
  sleep 10
done
sleep 10

echo "=== probe_norm_ulp $(date -u +%H:%M:%S)"
"$PY" -u scripts/probe_norm_ulp.py 2>&1 | tail -12
echo "=== probe4 done $(date -u +%H:%M:%S)"

#!/bin/bash
# Conv weight-format probe, then the six-step memory check with the current tree. Inside nzf-cai-dev on node6.
set -uo pipefail

LOCAL=/mnt/disk0/jiangzs/nzf
BENCH=$LOCAL/bench_online
REPO=$LOCAL/ColossalAI
PY=$LOCAL/cai-npu/bin/python
CKPT=/mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID

export PYTHONPATH=$REPO
cd "$REPO/examples/worldmodel/cosmos3" || exit 1

echo "=== probe_conv_transdata $(date -u +%H:%M:%S)"
"$PY" -u scripts/probe_conv_transdata.py --checkpoint "$CKPT" --iters 10 2>&1 | tail -60

echo "=== memcheck $(date -u +%H:%M:%S)"
MASTER_PORT=29895 bash scripts/bench_online_step.sh memcheck 32 6 \
  --max-episodes 64 --action-cache "$LOCAL/action_cache" --worker-threads 4 --vae-content-pad 16 \
  > "$BENCH/logs/memcheck.log" 2>&1
echo "memcheck exit $? $(date -u +%H:%M:%S)"
echo "=== conv_and_memcheck done $(date -u +%H:%M:%S)"

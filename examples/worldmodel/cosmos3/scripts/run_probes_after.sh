#!/bin/bash
# Wait for run_followups.sh, then run the two layout probes. Inside the nzf-cai-dev container.
set -uo pipefail

LOCAL=/mnt/disk0/jiangzs/nzf
BENCH=$LOCAL/bench_online
REPO=$LOCAL/ColossalAI
PY=$LOCAL/cai-npu/bin/python
CKPT=/mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID

export PYTHONPATH=$REPO
cd "$REPO/examples/worldmodel/cosmos3" || exit 1

for _ in $(seq 1 240); do
  if grep -qa 'followups done' "$BENCH/logs/followups.log" 2>/dev/null; then break; fi
  sleep 10
done
sleep 15

echo "=== probe_conv_transdata $(date -u +%H:%M:%S)"
"$PY" -u scripts/probe_conv_transdata.py --checkpoint "$CKPT" 2>&1 | tail -70

echo "=== probe_layout_ops $(date -u +%H:%M:%S)"
"$PY" -u scripts/probe_layout_ops.py --checkpoint "$CKPT" 2>&1 | tail -40

echo "=== probes done $(date -u +%H:%M:%S)"

#!/bin/bash
# After the first probe batch: the three kernel-candidate probes. Inside the nzf-cai-dev container.
set -uo pipefail

LOCAL=/mnt/disk0/jiangzs/nzf
BENCH=$LOCAL/bench_online
REPO=$LOCAL/ColossalAI
PY=$LOCAL/cai-npu/bin/python
CKPT=/mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID

export PYTHONPATH=$REPO
cd "$REPO/examples/worldmodel/cosmos3" || exit 1

for _ in $(seq 1 360); do
  if grep -qa 'probes done' "$BENCH/logs/probes.log" 2>/dev/null; then break; fi
  sleep 10
done
sleep 15

echo "=== probe_avgdown_equiv $(date -u +%H:%M:%S)"
"$PY" -u scripts/probe_avgdown_equiv.py --checkpoint "$CKPT" 2>&1 | tail -30

echo "=== probe_nemotron_norm $(date -u +%H:%M:%S)"
"$PY" -u scripts/probe_nemotron_norm.py 2>&1 | tail -40

echo "=== probe_dit_norm $(date -u +%H:%M:%S)"
"$PY" -u scripts/probe_dit_norm.py 2>&1 | tail -30

echo "=== probes2 done $(date -u +%H:%M:%S)"

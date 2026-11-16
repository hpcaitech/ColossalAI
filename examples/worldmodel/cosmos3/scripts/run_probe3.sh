#!/bin/bash
# Re-run the conv weight-format probe after the other probes (it needs torch_npu.profiler, not torch.npu.profiler).
set -uo pipefail

LOCAL=/mnt/disk0/jiangzs/nzf
BENCH=$LOCAL/bench_online
REPO=$LOCAL/ColossalAI
PY=$LOCAL/cai-npu/bin/python
CKPT=/mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID

export PYTHONPATH=$REPO
cd "$REPO/examples/worldmodel/cosmos3" || exit 1

for _ in $(seq 1 240); do
  if grep -qa 'probes2 done' "$BENCH/logs/probes2.log" 2>/dev/null; then break; fi
  sleep 10
done
sleep 10

echo "=== probe_conv_transdata $(date -u +%H:%M:%S)"
"$PY" -u scripts/probe_conv_transdata.py --checkpoint "$CKPT" --iters 10 2>&1 | tail -50
echo "=== probe3 done $(date -u +%H:%M:%S)"

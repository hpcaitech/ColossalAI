#!/bin/bash
# Production-scale check of the combined data/VAE fixes: 32 samples/rank, 64 episodes, 6 steps.
#
#   bash examples/worldmodel/cosmos3/scripts/run_integrated.sh [node]
#
# Flags: --action-cache (bit-exact parquet removal), --worker-threads 4 (restore worker parallelism),
# --vae-content-pad 16 (aggressive canvas crop; swap to 64 for the conservative gate).
set -uo pipefail

NODE=${1:-node6}
SSH=(ssh -o ConnectTimeout=15 "$NODE")
BENCH=/mnt/disk0/jiangzs/nzf/bench_online
STEP=/mnt/disk0/jiangzs/nzf/ColossalAI/examples/worldmodel/cosmos3/scripts/bench_online_step.sh
SUM=/mnt/disk0/jiangzs/nzf/ColossalAI/examples/worldmodel/cosmos3/scripts/summarize_online_metrics.py
PY=/mnt/disk0/jiangzs/nzf/cai-npu/bin/python
TAG=integrated
PORT=29844
PAD=${CANVAS_PAD:-16}

"${SSH[@]}" "rm -f $BENCH/logs/$TAG.log"
echo "=== $TAG $(date -u +%H:%M:%S) pad=$PAD"
"${SSH[@]}" "setsid nohup docker exec -e MASTER_PORT=$PORT nzf-cai-dev bash $STEP $TAG 32 6 --max-episodes 64 \
  --action-cache /mnt/disk0/jiangzs/nzf/action_cache --worker-threads 4 --vae-content-pad $PAD \
  > $BENCH/logs/$TAG.log 2>&1 < /dev/null & echo started"
for _ in $(seq 1 90); do
  hit=$("${SSH[@]}" "grep -qaE '(^| )done\$|Traceback|RuntimeError' $BENCH/logs/$TAG.log 2>/dev/null && echo yes" 2>/dev/null || true)
  [ "$hit" = yes ] && break
  sleep 10
done
"${SSH[@]}" "docker exec nzf-cai-dev pkill -f train_sft.py 2>/dev/null || true; \
  docker exec nzf-cai-dev pkill -f torch.distributed.run 2>/dev/null || true"
sleep 2
"${SSH[@]}" "grep -aE '\[step |done' $BENCH/logs/$TAG.log | tail -14"
"${SSH[@]}" "docker exec nzf-cai-dev bash -c 'cd /mnt/disk0/jiangzs/nzf/ColossalAI && PYTHONPATH=. $PY $SUM \
  --metrics-dir $BENCH/runs/$TAG/metrics --world-size 8'"
echo "INTEGRATED_DONE $(date -u +%H:%M:%S)"

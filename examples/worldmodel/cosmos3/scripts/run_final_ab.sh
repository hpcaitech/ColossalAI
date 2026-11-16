#!/bin/bash
# Best-configuration check: fused norm kernel on both arms; the second arm adds the uint8 data path with the
# rewritten fused normalisation kernel (COSMOS3_VIDEO_NORM_KERNEL=1).
#
#   bash examples/worldmodel/cosmos3/scripts/run_final_ab.sh [node]
set -uo pipefail

NODE=${1:-node6}
SSH=(ssh -o ConnectTimeout=15 "$NODE")
BENCH=/mnt/disk0/jiangzs/nzf/bench_online
STEP=/mnt/disk0/jiangzs/nzf/ColossalAI/examples/worldmodel/cosmos3/scripts/bench_online_step.sh
SUM=/mnt/disk0/jiangzs/nzf/ColossalAI/examples/worldmodel/cosmos3/scripts/summarize_online_metrics.py
PY=/mnt/disk0/jiangzs/nzf/cai-npu/bin/python
SAMPLES=${SAMPLES_PER_RANK:-32}
ITERS=${ITERS:-5}

run() {
  local tag=$1; shift
  local envs=$1; shift
  local port=$((29901 + RANDOM % 300))
  "${SSH[@]}" "mkdir -p $BENCH/logs $BENCH/runs; rm -f $BENCH/logs/$tag.log; \
    docker exec nzf-cai-dev bash -lc 'rm -rf $BENCH/runs/$tag' 2>/dev/null || true"
  echo "=== $tag $(date -u +%H:%M:%S) env=[$envs] args=$*"
  "${SSH[@]}" "setsid nohup docker exec -e MASTER_PORT=$port $envs nzf-cai-dev bash $STEP $tag $SAMPLES $ITERS \
    --max-episodes 64 --action-cache /mnt/disk0/jiangzs/nzf/action_cache --worker-threads 4 \
    --vae-content-pad 16 $* > $BENCH/logs/$tag.log 2>&1 < /dev/null & echo started"
  local hit=""
  for _ in $(seq 1 200); do
    hit=$("${SSH[@]}" "grep -qaE '(^| )done\$|Traceback|RuntimeError' $BENCH/logs/$tag.log 2>/dev/null && echo yes" 2>/dev/null || true)
    [ "$hit" = yes ] && break
    sleep 10
  done
  "${SSH[@]}" "docker exec nzf-cai-dev pkill -f train_sft.py 2>/dev/null || true; \
    docker exec nzf-cai-dev pkill -f torch.distributed.run 2>/dev/null || true"
  sleep 2
  "${SSH[@]}" "grep -aE '\[step |done' $BENCH/logs/$tag.log | tail -6"
  "${SSH[@]}" "docker exec nzf-cai-dev bash -c 'cd /mnt/disk0/jiangzs/nzf/ColossalAI && PYTHONPATH=. $PY $SUM \
    --metrics-dir $BENCH/runs/$tag/metrics --world-size 8'"
}

run final_float ""
run final_uint8k "-e COSMOS3_VIDEO_NORM_KERNEL=1" --video-uint8

echo "FINAL_AB_DONE $(date -u +%H:%M:%S)"

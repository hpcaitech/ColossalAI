#!/bin/bash
# ZeRO gradient-bucket A/B: 12M elements (default, ~119 buckets / 56 reduce-scatters) versus 48M (~30 buckets).
#
# The profiled step puts ~10.5 s of hcom against a fabric that does a 512 MiB reduce-scatter in 4.2 ms, so the
# question is whether the cost is per-bucket sync overhead (bigger buckets win) or contention (they do not).
#
#   bash examples/worldmodel/cosmos3/scripts/run_bucket_ab.sh [node]
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
  local port=$((29701 + RANDOM % 300))
  "${SSH[@]}" "mkdir -p $BENCH/logs $BENCH/runs; rm -f $BENCH/logs/$tag.log; \
    docker exec nzf-cai-dev bash -lc 'rm -rf $BENCH/runs/$tag' 2>/dev/null || true"
  echo "=== $tag $(date -u +%H:%M:%S) args=$*"
  "${SSH[@]}" "setsid nohup docker exec -e MASTER_PORT=$port nzf-cai-dev bash $STEP $tag $SAMPLES $ITERS \
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

run bucket12
run bucket48 --zero-bucket-size-m 48

echo "BUCKET_AB_DONE $(date -u +%H:%M:%S)"

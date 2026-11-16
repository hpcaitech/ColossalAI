#!/bin/bash
# Two-case A/B after the full-chain races: a fresh log per case, a unique rendezvous port, and a pkill after `done`.
#
#   bash examples/worldmodel/cosmos3/scripts/ab_targeted.sh [node]
#
# Cases: wt4 (--worker-threads 4) and canvas16 (--vae-content-pad 16), at 8 samples/rank over 8 episodes.
set -uo pipefail

NODE=${1:-node6}
SSH=(ssh -o ConnectTimeout=15 "$NODE")
BENCH=/mnt/disk0/jiangzs/nzf/bench_online
STEP=/mnt/disk0/jiangzs/nzf/ColossalAI/examples/worldmodel/cosmos3/scripts/bench_online_step.sh
SUM=/mnt/disk0/jiangzs/nzf/ColossalAI/examples/worldmodel/cosmos3/scripts/summarize_online_metrics.py
PY=/mnt/disk0/jiangzs/nzf/cai-npu/bin/python
SAMPLES=${SAMPLES_PER_RANK:-8}
ITERS=${ITERS:-4}

run() {
  local tag=$1; shift
  local port=$((29531 + RANDOM % 500))
  "${SSH[@]}" "rm -f $BENCH/logs/$tag.log"
  echo "=== $tag $(date -u +%H:%M:%S) port=$port args=$*"
  "${SSH[@]}" "setsid nohup docker exec -e MASTER_PORT=$port nzf-cai-dev bash $STEP $tag $SAMPLES $ITERS \
    --max-episodes 8 --action-cache /mnt/disk0/jiangzs/nzf/action_cache $* \
    > $BENCH/logs/$tag.log 2>&1 < /dev/null & echo started"
  local hit=""
  for _ in $(seq 1 60); do
    hit=$("${SSH[@]}" "grep -qaE '(^| )done\$|Traceback|RuntimeError' $BENCH/logs/$tag.log 2>/dev/null && echo yes" 2>/dev/null || true)
    [ "$hit" = yes ] && break
    sleep 8
  done
  "${SSH[@]}" "docker exec nzf-cai-dev pkill -f train_sft.py 2>/dev/null || true; \
    docker exec nzf-cai-dev pkill -f torch.distributed.run 2>/dev/null || true"
  sleep 2
  "${SSH[@]}" "grep -aE '\[step |done' $BENCH/logs/$tag.log | tail -8"
  "${SSH[@]}" "docker exec nzf-cai-dev bash -c 'cd /mnt/disk0/jiangzs/nzf/ColossalAI && PYTHONPATH=. $PY $SUM \
    --metrics-dir $BENCH/runs/$tag/metrics --world-size 8'"
}

run wt4 --worker-threads 4
run canvas16 --vae-content-pad 16
echo "TARGETED_DONE $(date -u +%H:%M:%S)"

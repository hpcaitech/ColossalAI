#!/bin/bash
# Does the fused-norm win survive two nodes? node1 = rank 0 (master 192.168.0.46), node6 = rank 1.
# Two arms, three steps each; the only variable is --vae-norm-kernel.
#
#   bash examples/worldmodel/cosmos3/scripts/run_2node_ab.sh
set -uo pipefail

BENCH=/mnt/disk0/jiangzs/nzf/bench_online
STEP2=/mnt/disk0/jiangzs/nzf/ColossalAI/examples/worldmodel/cosmos3/scripts/bench_online_step_2node.sh
SUM=/mnt/disk0/jiangzs/nzf/ColossalAI/examples/worldmodel/cosmos3/scripts/summarize_online_metrics.py
PY=/mnt/disk0/jiangzs/nzf/cai-npu/bin/python
MASTER=192.168.0.46
SAMPLES=${SAMPLES_PER_RANK:-32}
ITERS=${ITERS:-3}

n1() { ssh -o ConnectTimeout=15 node1 "$@"; }
n6() { ssh -o ConnectTimeout=15 node6 "$@"; }

run() {
  local tag=$1; shift
  local port=$((30101 + RANDOM % 200))
  for node in node1 node6; do
    ssh -o ConnectTimeout=15 "$node" "mkdir -p $BENCH/logs; rm -f $BENCH/logs/$tag.$( [ $node = node1 ] && echo n1 || echo n6 ).log; \
      docker exec nzf-cai-dev bash -lc 'rm -rf $BENCH/runs/$tag' 2>/dev/null || true"
  done
  echo "=== $tag $(date -u +%H:%M:%S) args=$*"
  n1 "setsid nohup docker exec -e MASTER_PORT=$port nzf-cai-dev bash $STEP2 $tag $SAMPLES $ITERS 0 $MASTER $* \
      > $BENCH/logs/$tag.n1.log 2>&1 < /dev/null & echo n1-started"
  sleep 20
  n6 "setsid nohup docker exec -e MASTER_PORT=$port nzf-cai-dev bash $STEP2 $tag $SAMPLES $ITERS 1 $MASTER $* \
      > $BENCH/logs/$tag.n6.log 2>&1 < /dev/null & echo n6-started"
  local hit=""
  for _ in $(seq 1 240); do
    hit=$(n1 "grep -qaE '(^| )done\$|Traceback|RuntimeError' $BENCH/logs/$tag.n1.log 2>/dev/null && echo y" 2>/dev/null || true)
    if [ "$hit" = y ]; then break; fi
    sleep 10
  done
  sleep 5
  for node in node1 node6; do
    ssh -o ConnectTimeout=15 "$node" "docker exec nzf-cai-dev pkill -f train_sft.py 2>/dev/null || true; \
      docker exec nzf-cai-dev pkill -f torch.distributed.run 2>/dev/null || true"
  done
  sleep 5
  for taglog in $tag.n1 $tag.n6; do
    echo "--- $taglog"
    ssh -o ConnectTimeout=15 "$( [[ $taglog == *.n1 ]] && echo node1 || echo node6 )" \
      "grep -aE '^\[step |^done' $BENCH/logs/$taglog.log | tail -5"
  done
}

run norm_off_2n --no-vae-norm-kernel
run norm_on_2n

echo "TWO_NODE_AB_DONE $(date -u +%H:%M:%S)"

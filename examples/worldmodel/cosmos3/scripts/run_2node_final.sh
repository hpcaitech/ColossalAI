#!/bin/bash
# One arm, three steps, two nodes: the current tree end to end (two-pass norm, AvgDown zero-pad skip, memory
# guard). node1 = rank 0 (192.168.0.46), node6 = rank 1.
#
#   bash examples/worldmodel/cosmos3/scripts/run_2node_final.sh [TAG] [STEPS]
set -uo pipefail

BENCH=/mnt/disk0/jiangzs/nzf/bench_online
STEP2=/mnt/disk0/jiangzs/nzf/ColossalAI/examples/worldmodel/cosmos3/scripts/bench_online_step_2node.sh
TAG=${1:-final_2n}
ITERS=${2:-3}
SAMPLES=${SAMPLES_PER_RANK:-32}
MASTER=192.168.0.46
PORT=$((30201 + RANDOM % 200))

n1() { ssh -o ConnectTimeout=15 node1 "$@"; }
n6() { ssh -o ConnectTimeout=15 node6 "$@"; }

for node in node1 node6; do
  ssh -o ConnectTimeout=15 "$node" "mkdir -p $BENCH/logs; rm -f $BENCH/logs/$TAG.$([ $node = node1 ] && echo n1 || echo n6).log; \
    docker exec nzf-cai-dev bash -lc 'rm -rf $BENCH/runs/$TAG' 2>/dev/null || true"
done

echo "=== $TAG $(date -u +%H:%M:%S) steps=$ITERS"
n1 "setsid nohup docker exec -e MASTER_PORT=$PORT nzf-cai-dev bash $STEP2 $TAG $SAMPLES $ITERS 0 $MASTER \
    > $BENCH/logs/$TAG.n1.log 2>&1 < /dev/null & echo n1-started"
sleep 20
n6 "setsid nohup docker exec -e MASTER_PORT=$PORT nzf-cai-dev bash $STEP2 $TAG $SAMPLES $ITERS 1 $MASTER \
    > $BENCH/logs/$TAG.n6.log 2>&1 < /dev/null & echo n6-started"

for _ in $(seq 1 240); do
  if n1 "grep -qaE '(^| )done\$|Traceback|RuntimeError' $BENCH/logs/$TAG.n1.log 2>/dev/null && echo y" 2>/dev/null | grep -q y; then
    break
  fi
  sleep 10
done
sleep 5
for node in node1 node6; do
  ssh -o ConnectTimeout=15 "$node" "docker exec nzf-cai-dev pkill -f train_sft.py 2>/dev/null || true; \
    docker exec nzf-cai-dev pkill -f torch.distributed.run 2>/dev/null || true"
done
sleep 5
for taglog in $TAG.n1 $TAG.n6; do
  echo "--- $taglog"
  ssh -o ConnectTimeout=15 "$([[ $taglog == *.n1 ]] && echo node1 || echo node6)" \
    "docker exec nzf-cai-dev bash -lc \"grep -aE '^\[step [0-9]+\].*\| step |^done' $BENCH/logs/$taglog.log | tail -4\""
done

echo "FINAL_2N_DONE $(date -u +%H:%M:%S)"

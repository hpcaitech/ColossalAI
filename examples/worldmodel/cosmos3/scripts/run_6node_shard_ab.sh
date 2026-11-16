#!/bin/bash
# The six-node shard A/B, ready to run the moment node5/7/8/9 are free.
#
#   NODES="node1 node5 node6 node7 node8 node9" MASTER=192.168.0.46 bash run_6node_shard_ab.sh
#
# Two arms, three steps each, same everything except --hsdp-shard-size:
#   shard 8  -> replica 6 across the nodes (the current default shape)
#   shard 48 -> pure ZeRO, no replica group at all
#
# What the 16-rank runs already establish, so the reader knows what this can and cannot show:
#   * cross-node collectives lose only 5-7% of bandwidth (reduce_scatter 119 -> 112.65 GiB/s), and one step's
#     gradients are worth ~10 ms of all-reduce, while the profile's exposed communication is 1.3-1.9 s -- the
#     multi-node cost is synchronisation and stragglers, not bytes;
#   * shard 8 vs 16 at two nodes was a wash (24.94/27.43 vs 24.73/27.16 s) with 2 GiB less memory for shard=world.
#   So expect a small effect here; this run is to confirm it does not invert at six nodes, not to find a big lever.
#
# The arm is accepted only if the step median moves by more than the +/-0.4 s run-to-run noise measured at 16 ranks.
set -uo pipefail

BENCH=/mnt/disk0/jiangzs/nzf/bench_online
STEP2=/mnt/disk0/jiangzs/nzf/ColossalAI/examples/worldmodel/cosmos3/scripts/bench_online_step_2node.sh
ITERS=${ITERS:-3}
SAMPLES=${SAMPLES_PER_RANK:-32}
NODES=${NODES:-}
MASTER=${MASTER:-}
PORT_BASE=${PORT_BASE:-30300}

if [ -z "$NODES" ] || [ -z "$MASTER" ]; then
  echo "usage: NODES=\"node1 node5 node6 node7 node8 node9\" MASTER=<rank0 bond1 ip> bash $0" >&2
  echo "       (node5/7/8/9 must be free; do not run this while they are in another test)" >&2
  exit 2
fi

read -r -a NODE_LIST <<<"$NODES"
WORLD=$(( ${#NODE_LIST[@]} * 8 ))

run_arm() {
  local tag=$1 shard=$2 port=$(( PORT_BASE + RANDOM % 200 ))
  local rank=0
  for node in "${NODE_LIST[@]}"; do
    ssh -o ConnectTimeout=15 "$node" "mkdir -p $BENCH/logs; rm -f $BENCH/logs/$tag.$node.log; \
      docker exec nzf-cai-dev bash -lc 'rm -rf $BENCH/runs/$tag' 2>/dev/null || true"
    ssh -o ConnectTimeout=15 "$node" "setsid nohup docker exec -e MASTER_PORT=$port nzf-cai-dev bash $STEP2 \
        $tag $SAMPLES $ITERS $rank $MASTER --max-episodes 2048 \
        --action-cache /mnt/disk0/jiangzs/nzf/action_cache --worker-threads 4 --vae-content-pad 16 \
        --hsdp-shard-size $shard > $BENCH/logs/$tag.$node.log 2>&1 < /dev/null & echo $node-started"
    rank=$((rank + 1))
    sleep 15
  done

  for _ in $(seq 1 360); do
    if ssh -o ConnectTimeout=15 "${NODE_LIST[0]}" \
        "grep -qaE '(^| )done\$|Traceback|RuntimeError|OutOfMemory' $BENCH/logs/$tag.${NODE_LIST[0]}.log 2>/dev/null && echo y" 2>/dev/null | grep -q y; then
      break
    fi
    sleep 10
  done
  sleep 5
  for node in "${NODE_LIST[@]}"; do
    ssh -o ConnectTimeout=15 "$node" "docker exec nzf-cai-dev pkill -f train_sft.py 2>/dev/null; \
      docker exec nzf-cai-dev pkill -f torch.distributed.run 2>/dev/null"
  done

  echo "=== $tag (shard $shard, $WORLD ranks)"
  ssh -o ConnectTimeout=15 "${NODE_LIST[0]}" "docker exec nzf-cai-dev bash -lc \"grep -aE '^\[step [0-9]+\].*\| step ' $BENCH/logs/$tag.${NODE_LIST[0]}.log | tail -3\""
}

echo "world $WORLD ranks over: ${NODE_LIST[*]} (master $MASTER)"
run_arm shard8_6n 8
run_arm shardworld_6n "$WORLD"
echo "SIX_NODE_SHARD_AB_DONE $(date -u +%H:%M:%S)"
echo "compare: step medians of steps 2..N on ${NODE_LIST[0]}; the +/-0.4 s noise floor applies"

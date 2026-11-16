#!/bin/bash
# Start the six-node training with the dashboard and the per-node system collectors, from the workstation.
#
#   NODES="node1 node5 node6 node7 node8 node9" MASTER=192.168.0.46 DASHBOARD=node6 bash run_6node_train.sh TAG
#
#   RESUME=<checkpoints dir>  restarts from a published generation: same weights, optimizer, step and sampler
#                       cursor. Point it at the *stopped* run's <run>/checkpoints (each node reads its own copy;
#                       format 3 publishes per node). Use a new TAG so the stopped run keeps its charts, or the
#                       same TAG to continue in place -- and note the out directory is then not wiped.
#
#   NUMA_PIN=1 ...      pins each rank's process tree to the socket its NPU sits on (numa_affinity.py). The map is
#                       derived per node from npu-smi + sysfs; the trainer logs one "[numa]" line per node.
#
# Rank 0 must be the first node in NODES (it is the torchrun master). Every node needs: the repo synced under
# /mnt/disk0/jiangzs/nzf/ColossalAI, the action cache under /mnt/disk0/jiangzs/nzf/action_cache, and the
# nzf-cai-dev container idle.
#
# Monitoring (see examples/worldmodel/cosmos3_monitor/README.md):
#   * one `collect` per node writes <out>/metrics/system/node-<host>.jsonl (CPU, memory, network, per-card HBM);
#   * one `serve` on $DASHBOARD exposes the dashboard; tunnel to it with the command printed at the end.
set -uo pipefail

TAG=${1:?tag}
NODES=${NODES:-}
MASTER=${MASTER:-}
DASHBOARD=${DASHBOARD:-node6}
LOCAL=/mnt/disk0/jiangzs/nzf
BENCH=$LOCAL/bench_online
OUT=$BENCH/runs/$TAG
STEP6=$LOCAL/ColossalAI/examples/worldmodel/cosmos3/scripts/bench_online_step_6node.sh
PORT=${MASTER_PORT:-30400}
DASH_PORT=${DASH_PORT:-18790}
PY=$LOCAL/cai-npu/bin/python

if [ -z "$NODES" ] || [ -z "$MASTER" ]; then
  echo "usage: NODES=\"node1 node5 node6 node7 node8 node9\" MASTER=<rank0 bond1 ip> bash $0 TAG" >&2
  exit 2
fi
read -r -a NODE_LIST <<<"$NODES"

# Refuse to start on a node that is already training, and say which.  The collector's command line also contains
# "--entry train_sft.py", so the match excludes it by name; otherwise a leftover collector makes every node look busy.
busy=$(for node in "${NODE_LIST[@]}"; do
  ssh -o ConnectTimeout=15 "$node" "docker exec nzf-cai-dev bash -lc 'ps -eo args | grep \"[t]rain_sft.py\" | grep -vc cosmos3_monitor'" 2>/dev/null
done | paste -sd, -)
if echo "$busy" | grep -qvE '^(0,)*0$'; then
  echo "refusing to start: per-node train_sft.py counts are [$busy] over ${NODE_LIST[*]}" >&2
  exit 3
fi

for node in "${NODE_LIST[@]}"; do
  if [ -n "${RESUME:-}" ]; then
    # Never wipe a directory a resume may be reading from; only make sure the logs exist.
    ssh -o ConnectTimeout=15 "$node" "mkdir -p $BENCH/logs" 2>/dev/null || true
  else
    ssh -o ConnectTimeout=15 "$node" "mkdir -p $BENCH/logs; docker exec nzf-cai-dev bash -lc 'rm -rf $OUT' 2>/dev/null || true"
  fi
done

echo "=== $TAG over ${#NODE_LIST[@]} nodes x 8 cards, master $MASTER, numa_pin=${NUMA_PIN:-0}, resume=${RESUME:-none}, $(date -u +%H:%M:%S)"
# NNODES must reach the step script: it defaults to 6 and torchrun would wait for two nodes that never join.
# CKPT/TRAIN_ARGS are optional: a warm-start checkpoint directory, and extra train_sft.py arguments.
DOCKER_ENV="-e MASTER_PORT=$PORT -e NUMA_PIN=${NUMA_PIN:-0} -e NNODES=${#NODE_LIST[@]}"
[ -n "${RESUME:-}" ] && DOCKER_ENV="$DOCKER_ENV -e RESUME=$RESUME"
[ -n "${CKPT:-}" ] && DOCKER_ENV="$DOCKER_ENV -e CKPT=$CKPT"
[ -n "${SAVE_EVERY:-}" ] && DOCKER_ENV="$DOCKER_ENV -e SAVE_EVERY=$SAVE_EVERY"
[ -n "${OMP_THREADS:-}" ] && DOCKER_ENV="$DOCKER_ENV -e OMP_THREADS=$OMP_THREADS"
rank=0
for node in "${NODE_LIST[@]}"; do
  # The redirect target has to be the path train_sft.py records as --log-path (TAG.n<rank>.log); otherwise
  # the dashboard's Logs tab looks for a file that nobody writes.
  ssh -o ConnectTimeout=15 "$node" "setsid nohup docker exec $DOCKER_ENV nzf-cai-dev bash $STEP6 \
      $TAG $rank $MASTER ${TRAIN_ARGS:-} > $BENCH/logs/$TAG.n$rank.log 2>&1 < /dev/null & echo $node-started"
  rank=$((rank + 1))
  sleep 12
done

# System collectors: one per node, writing into the shared run directory.
for node in "${NODE_LIST[@]}"; do
  ssh -o ConnectTimeout=15 "$node" "setsid nohup docker exec nzf-cai-dev bash -lc 'cd $LOCAL/ColossalAI && \
      PYTHONPATH=. $PY -m examples.worldmodel.cosmos3_monitor collect --run-dir $OUT --interval 5 \
      --entry train_sft.py' > $BENCH/logs/$TAG.collect.$node.log 2>&1 < /dev/null & echo $node-collector"
done

# Dashboard on one node; it only reads the shared run directory.
ssh -o ConnectTimeout=15 "$DASHBOARD" "setsid nohup docker exec nzf-cai-dev bash -lc 'cd $LOCAL/ColossalAI && \
    PYTHONPATH=. $PY -m examples.worldmodel.cosmos3_monitor serve --roots $OUT \
    --host 0.0.0.0 --port $DASH_PORT' > $BENCH/logs/$TAG.serve.log 2>&1 < /dev/null & echo dashboard-started"

cat <<EOF

started. first step includes cold data fill and kernel compiles (~5-10 min).

  watch:   ssh ${NODE_LIST[0]} 'docker exec nzf-cai-dev tail -f $BENCH/logs/$TAG.n0.log' | grep -E '^\\[step|TRAIN6'
  tunnel:  ssh -N -L $DASH_PORT:127.0.0.1:$DASH_PORT $DASHBOARD     # then http://127.0.0.1:$DASH_PORT
  stop:    for n in ${NODE_LIST[*]}; do ssh \$n "docker exec nzf-cai-dev pkill -f train_sft.py"; done
EOF

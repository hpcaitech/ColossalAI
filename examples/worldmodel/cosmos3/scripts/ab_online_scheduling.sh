#!/bin/bash
# Run the single-node scheduling A/B set on node6 through the unified container.
#
# Every case uses the same 8-card, 32-sample/rank, 4-step shape and a small 8-episode subset, so the cold start
# and the periodic parquet-boundary stall do not dominate the comparison. Steps 2-4 are the steady state; the
# summary drops step 1.
#
#   bash examples/worldmodel/cosmos3/scripts/ab_online_scheduling.sh [node]
#
# Cases:
#   base    default (no overlap, 4 workers)
#   ag      --overlap-allgather
#   w8      --num-workers 8
#   w8o2    --num-workers 8 with OMP_NUM_THREADS=2 per process
#   ga4     --grad-accum 4 (per-sample cost after amortizing optimizer/all-gather)
set -uo pipefail

NODE=${1:-node6}
# 32 samples/rank matches production; 8 makes a case finish in ~2 min because the first-step cost scales with the
# 256-sample batch, not with the dataset size. Relative comparisons are the same.
SAMPLES=${SAMPLES_PER_RANK:-32}
ITERS=${ITERS:-4}
SSH=(ssh -o ConnectTimeout=15 "$NODE")
BENCH=/mnt/disk0/jiangzs/nzf/bench_online
STEP=/mnt/disk0/jiangzs/nzf/ColossalAI/examples/worldmodel/cosmos3/scripts/bench_online_step.sh
SUM=/mnt/disk0/jiangzs/nzf/ColossalAI/examples/worldmodel/cosmos3/scripts/summarize_online_metrics.py
PY=/mnt/disk0/jiangzs/nzf/cai-npu/bin/python

wait_log() {
  local log=$1
  for _ in $(seq 1 90); do
    local hit
    hit=$("${SSH[@]}" "grep -qaE '(^| )done\$|Traceback|RuntimeError|OutOfMemory' '$log' 2>/dev/null && echo yes" 2>/dev/null || true)
    [ "$hit" = yes ] && return 0
    sleep 10
  done
  return 1
}

run_case() {
  local tag=$1; shift
  local omp=${OMP_THREADS:-}
  # Every case gets its own rendezvous port: `done` is printed before torch_npu's teardown releases the last
  # process, so reusing 29531 makes the next case fail with EADDRINUSE.
  local port=$((29531 + (RANDOM % 500)))
  local envflag="-e MASTER_PORT=$port"
  [ -n "$omp" ] && envflag="$envflag -e OMP_THREADS=$omp"
  echo "=== $tag $(date -u +%H:%M:%S) args=$* omp=${omp:-default}"
  "${SSH[@]}" "setsid nohup docker exec $envflag nzf-cai-dev bash $STEP $tag $SAMPLES $ITERS --max-episodes 8 \
    --action-cache /mnt/disk0/jiangzs/nzf/action_cache $* \
    > $BENCH/logs/$tag.log 2>&1 < /dev/null & echo started"
  if ! wait_log "$BENCH/logs/$tag.log"; then
    echo "--- $tag did not finish; tail:"
    "${SSH[@]}" "tail -20 $BENCH/logs/$tag.log"
    return 1
  fi
  # The training loop prints `done` before teardown; kill what remains so the next case gets the NPUs and the port.
  "${SSH[@]}" "docker exec nzf-cai-dev pkill -f train_sft.py 2>/dev/null || true; \
    docker exec nzf-cai-dev pkill -f torch.distributed.run 2>/dev/null || true"
  sleep 2
  "${SSH[@]}" "docker exec nzf-cai-dev bash -c 'cd /mnt/disk0/jiangzs/nzf/ColossalAI && PYTHONPATH=. $PY $SUM --metrics-dir $BENCH/runs/$tag/metrics --world-size 8'"
}

run_case base
run_case wt4 --worker-threads 4
if [ -n "${CANVAS_PAD:-}" ]; then
  run_case "canvas${CANVAS_PAD}" --vae-content-pad "${CANVAS_PAD}"
fi
run_case ag --overlap-allgather
run_case w8 --num-workers 8
OMP_THREADS=2 run_case w8o2 --num-workers 8
run_case ga4 --grad-accum 4
echo "AB_CHAIN_DONE $(date -u +%H:%M:%S)"

#!/bin/bash
# Long-horizon loss A/B for the canvas crop gate: full 736 vs pad64 (704) vs pad16 (656), 200 optimizer steps each.
#
#   bash examples/worldmodel/cosmos3/scripts/run_canvas_loss_ab.sh [node]
#
# Same 64 episodes, same seed, same action cache and worker threads (4); the only variable is --vae-content-pad.
# ~1.7 h per arm, ~5 h total. `done` is printed before teardown, hence the per-arm pkill and the unique port.
set -uo pipefail

NODE=${1:-node6}
SSH=(ssh -o ConnectTimeout=15 "$NODE")
BENCH=/mnt/disk0/jiangzs/nzf/bench_online
STEP=/mnt/disk0/jiangzs/nzf/ColossalAI/examples/worldmodel/cosmos3/scripts/bench_online_step.sh
SUM=/mnt/disk0/jiangzs/nzf/ColossalAI/examples/worldmodel/cosmos3/scripts/summarize_online_metrics.py
PY=/mnt/disk0/jiangzs/nzf/cai-npu/bin/python
STEPS=${STEPS:-200}

run() {
  local tag=$1; shift
  local port=$((29531 + RANDOM % 500))
  "${SSH[@]}" "rm -f $BENCH/logs/$tag.log"
  echo "=== $tag $(date -u +%H:%M:%S) pad_args=$*"
  "${SSH[@]}" "setsid nohup docker exec -e MASTER_PORT=$port nzf-cai-dev bash $STEP $tag 32 $STEPS \
    --max-episodes 64 --action-cache /mnt/disk0/jiangzs/nzf/action_cache --worker-threads 4 $* \
    > $BENCH/logs/$tag.log 2>&1 < /dev/null & echo started"
  local hit=""
  for _ in $(seq 1 1200); do
    hit=$("${SSH[@]}" "grep -qaE '(^| )done\$|Traceback|RuntimeError' $BENCH/logs/$tag.log 2>/dev/null && echo yes" 2>/dev/null || true)
    [ "$hit" = yes ] && break
    sleep 10
  done
  "${SSH[@]}" "docker exec nzf-cai-dev pkill -f train_sft.py 2>/dev/null || true; \
    docker exec nzf-cai-dev pkill -f torch.distributed.run 2>/dev/null || true"
  sleep 2
  "${SSH[@]}" "grep -aE '\[step |done' $BENCH/logs/$tag.log | tail -5"
  "${SSH[@]}" "docker exec nzf-cai-dev bash -c 'cd /mnt/disk0/jiangzs/nzf/ColossalAI && PYTHONPATH=. $PY $SUM \
    --metrics-dir $BENCH/runs/$tag/metrics --world-size 8'"
}

run canvasfull
run canvas64 --vae-content-pad 64
run canvas16 --vae-content-pad 16

echo "=== loss means (last 50 steps)"
for tag in canvasfull canvas64 canvas16; do
  "${SSH[@]}" "echo -n '$tag '; grep -aE '^\[step [0-9]+\] loss' $BENCH/logs/$tag.log | tail -50 | awk '{s+=\$4; n++} END {printf \"%.6f (%d steps)\n\", s/n, n}'"
done
echo "CANVAS_AB_DONE $(date -u +%H:%M:%S)"

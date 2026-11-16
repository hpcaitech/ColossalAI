#!/bin/bash
# Sequential follow-ups after the two-pass measurement: layout probes, the one-pass arm, and a steady-step profile.
# Runs *inside* the nzf-cai-dev container. Everything is logged to bench_online/logs/followups.log.
set -uo pipefail

LOCAL=/mnt/disk0/jiangzs/nzf
BENCH=$LOCAL/bench_online
REPO=$LOCAL/ColossalAI
PY=$LOCAL/cai-npu/bin/python
CKPT=/mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID
LOG=$BENCH/logs/followups.log

export PYTHONPATH=$REPO
export COLOSSALAI_EXTENSION_CACHE=$LOCAL/ext_cache
cd "$REPO/examples/worldmodel/cosmos3" || exit 1

stage() { echo "=== $1 $(date -u +%H:%M:%S)"; }

stage "wait for twopass"
for _ in $(seq 1 240); do
  if grep -qaE '^done$|Traceback' "$BENCH/logs/twopass.log" 2>/dev/null; then break; fi
  sleep 5
done
sleep 15

stage "probe_conv_transdata"
"$PY" -u scripts/probe_conv_transdata.py --checkpoint "$CKPT" 2>&1 | tail -70

stage "probe_layout_ops"
"$PY" -u scripts/probe_layout_ops.py --checkpoint "$CKPT" 2>&1 | tail -40

stage "one-pass arm"
MASTER_PORT=29891 COSMOS3_NORM_TWOPASS=0 bash scripts/bench_online_step.sh twopass_off 32 5 \
  --max-episodes 64 --action-cache "$LOCAL/action_cache" --worker-threads 4 --vae-content-pad 16 \
  > "$BENCH/logs/twopass_off.log" 2>&1
echo "one-pass exit $?"

stage "steady-step profile"
MASTER_PORT=29892 bash scripts/bench_online_step.sh stepprof 32 3 \
  --max-episodes 64 --action-cache "$LOCAL/action_cache" --worker-threads 4 --vae-content-pad 16 --profile-step 2 \
  > "$BENCH/logs/stepprof.log" 2>&1
echo "profile exit $?"

stage "followups done"

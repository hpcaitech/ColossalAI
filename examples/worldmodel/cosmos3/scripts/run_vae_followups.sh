#!/bin/bash
# Follow-ups after the fused-norm A/B, in one go on an otherwise idle node:
#   1. conv shapes + per-shape MFU        -> is the remaining 61% (conv) worth writing a kernel for?
#   2. production profile with the fused norm -> what the encoder's kernel mix looks like now
#   3. channels_last_3d profile           -> does the layout remove the conv-internal TransData?
#
#   bash examples/worldmodel/cosmos3/scripts/run_vae_followups.sh [node]
set -uo pipefail

NODE=${1:-node6}
SSH=(ssh -o ConnectTimeout=20 "$NODE")
BENCH=/mnt/disk0/jiangzs/nzf/bench_online
REPO=/mnt/disk0/jiangzs/nzf/ColossalAI
PY=/mnt/disk0/jiangzs/nzf/cai-npu/bin/python
CKPT=/mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID

run() {
  echo "=== $1 $(date -u +%H:%M:%S)"
  "${SSH[@]}" "docker exec nzf-cai-dev bash -lc 'cd $REPO && PYTHONPATH=. $PY $2'" 2>&1 | grep -vE "Authorized|^$"
}

run "conv-shapes" "examples/worldmodel/cosmos3/scripts/probe_conv_shapes.py --checkpoint $CKPT --batch 8 --timed"
run "profile-norm" "examples/worldmodel/cosmos3/scripts/profile_vae_deep.py --checkpoint $CKPT --out $BENCH/vae_deep_norm --batch 8 --dtype bfloat16 --profile --iters 3 --norm-kernel"
run "profile-channels-last" "examples/worldmodel/cosmos3/scripts/profile_vae_deep.py --checkpoint $CKPT --out $BENCH/vae_deep_cl --batch 8 --dtype bfloat16 --profile --iters 3 --channels-last"

for tag in vae_deep_norm vae_deep_cl; do
  echo "=== kernels $tag"
  "${SSH[@]}" "docker exec nzf-cai-dev bash -lc 'cd $REPO && PYTHONPATH=. $PY examples/worldmodel/cosmos3/scripts/analyze_vae_kernels.py $BENCH/$tag/contiguous-bfloat16-fp32 --top 12'" 2>&1 | grep -vE "Authorized|^$"
done
echo "VAE_FOLLOWUPS_DONE $(date -u +%H:%M:%S)"

#!/bin/bash
# Six-node (48 rank) training entry, run *inside* the nzf-cai-dev container on each node.
#
#   bench_online_step_6node.sh TAG NODE_RANK MASTER [train_sft args...]
#
# Same environment and the same optimisation set as the two-node benchmark, with three differences that matter for a
# real run rather than a 3-step A/B: the full dataset (no --max-episodes), the default 10000-step horizon, and a
# 100-step checkpoint interval so a crash costs at most an hour.
#
# Optimisations on by default (all verified bit-identical to upstream except where noted):
#   --vae-norm-kernel       fused channel-first RMSNorm, two-pass for wide shapes
#   --fast-vae-encoder      H/W padding into the conv, dead clone removed, AvgDown zero-pad skipped
#   --dit-sum-mean-norm     mean(-1) -> sum(-1)/H, removes a full-tensor division from the backward
#   COSMOS3_FAST_JITTER=1   ColorJitter's hue step through inductor (bit-identical, -33% per sample)
#   OMP_NUM_THREADS=4       leaves CPU for the augmentation workers (removes the periodic stall)
#   --vae-content-pad 16    canvas crop measured at relL2 1.48e-2, the same order as bf16 VAE noise
# Not enabled: --dit-norm-kernel, which is -3.3% step time but differs from the fp32 chain on ~1.2 elements per
# million by one ulp in the forward. Add it explicitly if that trade is accepted.
set -euo pipefail

TAG=${1:?tag}; NODE_RANK=${2:?}; MASTER=${3:?}; shift 3 || true
LOCAL=/mnt/disk0/jiangzs/nzf
# Weights. The default is the recipe checkpoint; a warm start points this at a directory assembled by
# scripts/prepare_warm_start.py (same layout, trained transformer/ inside).
CKPT=${CKPT:-/mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID}
# A published generation to restart from: weights AND optimizer, step and sampler cursor continue exactly where
# the stopped run left them.  Each node reads its own copy at this path (format 3 publishes per node); the
# checkpoint must have been written by this world size and with the same recipe fingerprint.
RESUME=${RESUME:-}
RESUME_ARGS=()
[ -n "$RESUME" ] && RESUME_ARGS=(--resume "$RESUME")
BENCH=$LOCAL/bench_online
PORT=${MASTER_PORT:-29531}
SAMPLES=${SAMPLES_PER_RANK:-32}
ITERS=${MAX_ITER:-10000}
SAVE_EVERY=${SAVE_EVERY:-100}
NODES=${NNODES:-6}

export PATH=$LOCAL/cai-npu/bin:$PATH
export PYTHONPATH=$LOCAL/ColossalAI:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-bond1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-900}
export COSMOS3_REPORT_PEAK_HBM=1
export COLOSSALAI_EXTENSION_CACHE=$LOCAL/ext_cache
# 4, not 8: eight OMP threads per rank plus 32 augmentation workers is exactly the 192 cores these nodes have, and
# the contention shows up as a periodic ~25 s stall.
export OMP_NUM_THREADS=${OMP_THREADS:-4}
# Compiled ColorJitter; the inductor cache must sit on local disk or every worker recompiles over NFS.
export COSMOS3_FAST_JITTER=${COSMOS3_FAST_JITTER:-1}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$LOCAL/ext_cache/inductor}
# NUMA_PIN=1 pins each rank's process tree (and its dataloader workers) to the socket its NPU is attached to; the map
# is derived from npu-smi and sysfs on this node, so it needs no per-machine table. Off by default.
# The `cd` matters: the container's default cwd is a stale NFS checkout of this repo that also has an `examples/`
# directory, and `python -m` puts cwd first on sys.path.
if [ "${NUMA_PIN:-0}" = "1" ]; then
  COSMOS3_NUMA_PIN=$(cd "$LOCAL/ColossalAI" && python -m examples.worldmodel.cosmos3.numa_affinity --print-map)
  if [ -z "$COSMOS3_NUMA_PIN" ]; then
    echo "NUMA_PIN=1 but the pin map came back empty" >&2
    exit 4
  fi
  export COSMOS3_NUMA_PIN
  echo "NUMA_PIN $TAG rank=$NODE_RANK map=$COSMOS3_NUMA_PIN"
fi
mkdir -p "$COLOSSALAI_EXTENSION_CACHE" "$TORCHINDUCTOR_CACHE_DIR" "$BENCH/runs/$TAG" "$BENCH/logs"

cd "$LOCAL/ColossalAI/examples/worldmodel/cosmos3"

echo "TRAIN6 $TAG rank=$NODE_RANK nodes=$NODES samples=$SAMPLES iters=$ITERS host=$(cat /proc/sys/kernel/hostname) master=$MASTER start $(date -u +%H:%M:%S) args=$*"
exec python -u -m torch.distributed.run \
  --nnodes="$NODES" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER" --master_port="$PORT" \
  --nproc_per_node=8 \
  train_sft.py \
  --checkpoint "$CKPT" \
  "${RESUME_ARGS[@]}" \
  --data /mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success \
  --out "$BENCH/runs/$TAG" \
  --samples-per-rank "$SAMPLES" --grad-accum 1 --hsdp-shard-size 8 \
  --max-iter "$ITERS" --save-every "$SAVE_EVERY" \
  --metrics-every 1 --log-every 1 \
  --num-workers 4 --action-cache "$LOCAL/action_cache" \
  --worker-threads 4 --image-augmentation torchvision \
  --vae-encode-batch 8 --fast-vae-encoder --vae-norm-kernel --dit-sum-mean-norm \
  --vae-content-pad 16 \
  --run-name "${RUN_NAME:-$TAG}" \
  --log-path "$BENCH/logs/$TAG.n$NODE_RANK.log" \
  "$@"

#!/bin/bash
# Standard single-node 8-card online benchmark, run *inside* the nzf-cai-dev container.
#
#   bench_online_step.sh TAG SAMPLES_PER_RANK MAX_ITER [train_sft args...]
#
# Everything lives on the node's local disk (repo, venv, extension cache, logs); the shared NFS is only used for
# the checkpoint and dataset, which are read-only. The caller redirects stdout to a log under
# /mnt/disk0/jiangzs/nzf/bench_online/logs and starts it with `setsid nohup ... < /dev/null`.
set -euo pipefail

TAG=${1:?tag}; SAMPLES=${2:-32}; ITERS=${3:-8}; shift 3 || true
LOCAL=/mnt/disk0/jiangzs/nzf
BENCH=$LOCAL/bench_online
PORT=${MASTER_PORT:-29531}

export PATH=$LOCAL/cai-npu/bin:$PATH
export PYTHONPATH=$LOCAL/ColossalAI:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-bond1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}
export COSMOS3_REPORT_PEAK_HBM=1
export COLOSSALAI_EXTENSION_CACHE=$LOCAL/ext_cache
# Compiled ColorJitter (F16.6): bit-identical to torchvision, -33% per sample, removes the periodic CPU stall.
# The inductor cache must sit on local disk, like the other extension caches, or every worker recompiles over NFS.
export COSMOS3_FAST_JITTER=${COSMOS3_FAST_JITTER:-1}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$LOCAL/ext_cache/inductor}
# 4, not 8: eight ranks x eight OMP threads plus 32 augmentation workers is exactly the 192 cores the node
# has, and the resulting CPU contention shows up as a periodic ~25 s stall (F16). Override with OMP_THREADS.
export OMP_NUM_THREADS=${OMP_THREADS:-${OMP_NUM_THREADS:-4}}
mkdir -p "$COLOSSALAI_EXTENSION_CACHE" "$BENCH/runs/$TAG" "$BENCH/logs"

cd "$LOCAL/ColossalAI/examples/worldmodel/cosmos3"

echo "BENCH $TAG samples=$SAMPLES iters=$ITERS host=$(cat /proc/sys/kernel/hostname) start $(date -u +%H:%M:%S) args=$*"
exec python -u -m torch.distributed.run \
  --nnodes=1 --node_rank=0 --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port="$PORT" \
  train_sft.py \
  --checkpoint /mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID \
  --data /mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success \
  --out "$BENCH/runs/$TAG" \
  --samples-per-rank "$SAMPLES" --grad-accum 1 --hsdp-shard-size 8 \
  --max-iter "$ITERS" --metrics-every 1 --log-every 1 \
  --num-workers 4 --max-episodes 64 --image-augmentation torchvision \
  --vae-encode-batch 8 --fast-vae-encoder --vae-norm-kernel --dit-sum-mean-norm \
  "$@"

#!/bin/bash
# Fingerprint the packed step from both data paths and diff the digests.
#
# A loss comparison cannot settle equivalence: the batch composition and the noise draw can differ between runs for
# reasons unrelated to the latents. The digests below cover the tensors that actually enter the forward
# (input_ids, position_ids) and the targets, so equal digests mean the two paths produced the same step.
#
#   eq_pack_hash.sh [PORT] [CACHE_DIR]
set -uo pipefail
PORT=${1:-29821}
CACHE=${2:-/mnt/disk0/jiangzs/nzf/latents/smoke_cache}
COMMON="--samples-per-rank 8 --grad-accum 1 --hsdp-shard-size 8 --max-iter 1 \
        --metrics-every 1 --log-every 1 --num-workers 4 --seed 42 --max-episodes 1 --image-augmentation none"

echo "=== ONLINE $(date +%H:%M:%S)"
bash /home/niuzifan/scripts/train_scale_inner.sh eq_hash_online 0 1 127.0.0.1 "$PORT" \
  $COMMON --vae-encode-batch 4 --fast-vae-encoder --pack-hash
echo "=== CACHED $(date +%H:%M:%S)"
bash /home/niuzifan/scripts/train_scale_inner.sh eq_hash_cached 0 1 127.0.0.1 "$PORT" \
  $COMMON --latent-cache "$CACHE" --pack-hash
echo "=== RESULT $(date +%H:%M:%S)"

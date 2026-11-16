#!/bin/bash
# Equivalence of the two training paths on the same windows: online encode vs cached latent.
#
# The cache was extracted with augmentation OFF, so for any window the cached latent should be the same tensor the
# online path would compute. Both runs then use the same seed and the same --max-episodes 1 selection, so the
# sampler visits the same windows in the same order and the only difference left is where the latent came from.
#
#   eq_online_vs_cached.sh [PORT] [CACHE_DIR]
set -uo pipefail
PORT=${1:-29801}
CACHE=${2:-/mnt/disk0/jiangzs/nzf/latents/smoke_cache}
COMMON="--samples-per-rank 8 --grad-accum 1 --hsdp-shard-size 8 --max-iter 3 \
        --metrics-every 1 --log-every 1 --num-workers 4 --seed 42 --max-episodes 1 --image-augmentation none"

echo "=== ONLINE $(date +%H:%M:%S)"
bash /home/niuzifan/scripts/train_scale_inner.sh eq_online 0 1 127.0.0.1 "$PORT" \
  $COMMON --vae-encode-batch 4 --fast-vae-encoder
echo "=== CACHED $(date +%H:%M:%S)"
bash /home/niuzifan/scripts/train_scale_inner.sh eq_cached 0 1 127.0.0.1 "$PORT" \
  $COMMON --latent-cache "$CACHE"
echo "=== SUMMARY $(date +%H:%M:%S)"
echo "-- online"
grep -hE "^\S+ \[step [0-9]+\] loss" /home/niuzifan/logs/eq_online.log 2>/dev/null | grep -E "loss [0-9]" | tail -n 6
echo "-- cached"
grep -hE "^\S+ \[step [0-9]+\] loss" /home/niuzifan/logs/eq_cached.log 2>/dev/null | grep -E "loss [0-9]" | tail -n 6

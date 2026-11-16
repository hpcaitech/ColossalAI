#!/bin/bash
# Step time with and without the DROID image augmentation, one node, 8 cards.
#
#   ab_augmentation_train.sh [SAMPLES_PER_RANK] [MASTER_PORT] [VAE_ENCODE_BATCH]
#
# Both cases use the same everything else, so the difference is the augmentation alone. Runs must not overlap
# with another training job on the node.
set -uo pipefail
SP_RANK=${1:-8}
PORT=${2:-29501}
VAE_BATCH=${3:-4}

for mode in torchvision none; do
  tag="aug-${mode}-s${SP_RANK}"
  echo "=== $tag $(date +%H:%M:%S)"
  rm -rf "/home/niuzifan/runs/scale-$tag"
  bash /home/niuzifan/scripts/train_scale_inner.sh "$tag" 0 1 127.0.0.1 "$PORT" \
    --samples-per-rank "$SP_RANK" --grad-accum 1 --hsdp-shard-size 8 --max-iter 4 \
    --metrics-every 1 --log-every 1 --num-workers 4 \
    --image-augmentation "$mode" --vae-encode-batch "$VAE_BATCH" --fast-vae-encoder
  echo "--- $tag end $(date +%H:%M:%S)"
done

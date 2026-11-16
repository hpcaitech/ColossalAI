#!/bin/bash
# A/B the window slicing of precompute_latents.py on one node: contiguous block vs interleaved stride,
# with and without decode workers. One process per card, 8 cards, same subset each time.
#
#   ab_latent_shard.sh [LIMIT] [MAX_EPISODES]
#
# MAX_EPISODES must stay small here. Without it the dataset builds a Window object per sample for the whole
# 20.11 M window set inside every one of the 8 processes, which is tens of GB each and gets the run OOM-killed
# before it encodes anything.
#
# Runs must not overlap with a training job: a stray train_sft.py left over from an earlier session keeps
# respawning its decode workers and starves the disk, which shows up as batches that never advance.
set -uo pipefail
LIMIT=${1:-512}
MAX_EPISODES=${2:-8}
INNER=/home/niuzifan/scripts/precompute_inner.sh
OUT_ROOT=/mnt/disk0/jiangzs/nzf/latents

run_case() {
  local tag=$1
  shift
  echo "=== $tag limit=$LIMIT episodes=$MAX_EPISODES $*"
  echo "--- $tag start $(date +%H:%M:%S)"
  rm -rf "$OUT_ROOT/$tag"
  bash "$INNER" "$tag" 0 8 --max-episodes "$MAX_EPISODES" --limit "$LIMIT" --encode-batch 8 "$@" \
    > "$OUT_ROOT/$tag.driver.log" 2>&1
  echo "--- $tag end $(date +%H:%M:%S)"
  grep -h "per_sample" "$OUT_ROOT/$tag"/rank-*.log | sed "s/^/$tag /"
}

run_case blk_nw0 --num-workers 0 --shard-mode block
run_case str_nw0 --num-workers 0 --shard-mode stride
run_case blk_nw4 --num-workers 4 --shard-mode block
run_case str_nw4 --num-workers 4 --shard-mode stride

echo "=== summary"
for tag in blk_nw0 str_nw0 blk_nw4 str_nw4; do
  files=$OUT_ROOT/$tag/rank-*.log
  samples=$(grep -h "per_sample" $files | sed -n 's/.*samples=\([0-9]*\).*/\1/p' | paste -sd+ - | bc)
  secs=$(grep -h "per_sample" $files | sed -n 's/.*seconds=\([0-9.]*\).*/\1/p' | paste -sd+ - | bc)
  ranks=$(grep -l "per_sample" $files | wc -l)
  echo "$tag ranks=$ranks samples=$samples mean_per_sample=$(echo "scale=3; $secs/$samples" | bc)"
done

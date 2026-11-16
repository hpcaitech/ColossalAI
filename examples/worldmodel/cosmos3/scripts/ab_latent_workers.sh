#!/bin/bash
# Sweep decode parallelism for latent pre-extraction on one node, 8 cards, contiguous block slicing.
#
#   ab_latent_workers.sh [WORKER_COUNTS] [ENCODE_BATCH] [LIMIT] [MAX_EPISODES]
#
# WORKER_COUNTS is comma separated ("2,4,8") rather than space separated: dev-container-node.sh forwards its
# arguments by joining them into one `bash -c` string, so a quoted "2 4 8" arrives as three separate argv
# entries and silently shifts every positional parameter after it.
#
# The first run in the sequence pays the import and VAE-load cost; later runs still pay import but the kernel
# caches are warm, so compare the per-sample column rather than the wall clock of each case.
set -uo pipefail
WORKER_COUNTS=$(printf '%s' "${1:-4,8}" | tr ',' ' ')
ENCODE_BATCH=${2:-4}
LIMIT=${3:-512}
MAX_EPISODES=${4:-8}
echo "effective worker_counts=[$WORKER_COUNTS] encode_batch=$ENCODE_BATCH limit=$LIMIT max_episodes=$MAX_EPISODES"
INNER=/home/niuzifan/scripts/precompute_inner.sh
OUT_ROOT=/mnt/disk0/jiangzs/nzf/latents

for workers in $WORKER_COUNTS; do
  tag="w${workers}_eb${ENCODE_BATCH}"
  echo "=== $tag $(date +%H:%M:%S)"
  rm -rf "$OUT_ROOT/$tag"
  bash "$INNER" "$tag" 0 8 --max-episodes "$MAX_EPISODES" --limit "$LIMIT" \
    --encode-batch "$ENCODE_BATCH" --num-workers "$workers" --shard-mode block \
    > "$OUT_ROOT/$tag.driver.log" 2>&1
  finished=$(grep -l "per_sample" "$OUT_ROOT/$tag"/rank-*.log 2>/dev/null | wc -l)
  echo "--- $tag done $(date +%H:%M:%S) ranks_reporting=$finished/8"
  grep -h "per_sample" "$OUT_ROOT/$tag"/rank-*.log 2>/dev/null | sed "s/^/$tag /"
  grep -l "OutOfMemoryError" "$OUT_ROOT/$tag"/rank-*.log 2>/dev/null | sed "s/^/$tag OOM /"
done

echo "=== summary"
for workers in $WORKER_COUNTS; do
  tag="w${workers}_eb${ENCODE_BATCH}"
  files=$OUT_ROOT/$tag/rank-*.log
  samples=$(grep -h "per_sample" $files 2>/dev/null | sed -n 's/.*samples=\([0-9]*\).*/\1/p' | paste -sd+ - | bc)
  secs=$(grep -h "per_sample" $files 2>/dev/null | sed -n 's/.*seconds=\([0-9.]*\).*/\1/p' | paste -sd+ - | bc)
  ranks=$(grep -l "per_sample" $files 2>/dev/null | wc -l)
  if [ -n "${samples:-}" ] && [ "$samples" -gt 0 ] && [ "$ranks" -gt 0 ]; then
    # Ranks run concurrently, so the node's throughput is the total sample count over the mean rank duration.
    mean_secs=$(echo "scale=6; $secs/$ranks" | bc)
    echo "$tag ranks=$ranks samples=$samples per_sample=$(echo "scale=3; $secs/$samples" | bc) samples_per_s_node=$(echo "scale=2; $samples/$mean_secs" | bc)"
  else
    echo "$tag ranks=0 (no completed ranks)"
  fi
done

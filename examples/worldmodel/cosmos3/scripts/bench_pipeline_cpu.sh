#!/bin/bash
# One node, 8 rank processes, CPU-only data-pipeline throughput. No NPU is touched: this measures what the host
# CPU can hand to the VAE, which is the ceiling the extractor and the trainer both sit under.
#
#   bench_pipeline_cpu.sh TAG WORKERS {on|off} [LIMIT]
set -uo pipefail
TAG=${1:?tag}; WORKERS=${2:-4}; AUG=${3:-on}; LIMIT=${4:-640}

LOCAL=/mnt/disk0/jiangzs/nzf
export PATH=$LOCAL/cai-npu/bin:${PATH}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTHONPATH=$LOCAL/ColossalAI:${PYTHONPATH:-}
DATA=/mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success
SCRIPT=$LOCAL/ColossalAI/examples/worldmodel/cosmos3/scripts/bench_pipeline_cpu.py
OUT=$LOCAL/latents/$TAG
mkdir -p "$OUT"

flag=""
[ "$AUG" = "on" ] && flag="--augment"
echo "CASE $TAG workers=$WORKERS augment=$AUG limit=$LIMIT host=$(cat /proc/sys/kernel/hostname) start $(date +%H:%M:%S)"

pids=()
for r in 0 1 2 3 4 5 6 7; do
  python -u "$SCRIPT" --data "$DATA" --rank $r --world 8 --workers "$WORKERS" --limit "$LIMIT" \
    --max-episodes 32 $flag > "$OUT/rank-$r.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

echo "CASE $TAG done $(date +%H:%M:%S)"
grep -h "per_sample" "$OUT"/rank-*.log | sed 's/^/  /'

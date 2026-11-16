#!/bin/bash
# Catch the periodic ~194 s data stall that shows up at the same update indices in every run.
#
# Samples /dev/shm usage, worker count, load and the per-worker CPU time while one screen runs, so the stall can be
# attributed instead of guessed at: shared-memory pressure and a worker that stopped consuming look different here.
set -uo pipefail

PROJECT=/mnt/disk0/jiangzs/nzf/infrarsi-cosmos3
TRAIN_LOG=/tmp/diag_screen.log
SAMPLE_LOG=/tmp/diag_samples.log
mkdir -p "$PROJECT"

cd "$PROJECT" || exit 1
rm -f "$TRAIN_LOG" "$SAMPLE_LOG"

STEPS=10 LOG=$TRAIN_LOG bash train.sh > /dev/null 2>&1 &
TRAIN_PID=$!

start=$(date +%s)
: > "$SAMPLE_LOG"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  now=$(date +%s)
  shm=$(df -m /dev/shm | awk 'NR==2 {print $3}')
  workers=$(pgrep -c -f spawn_main || true)
  load=$(cut -d' ' -f1 /proc/loadavg)
  # Total CPU seconds burned by the decode workers: flat means they are blocked, rising means they are working.
  worker_cpu=$(ps -eo times,args 2>/dev/null | grep "[s]pawn_main" | awk '{s+=$1} END {print s+0}')
  steps=$(grep -c "| samples" "$TRAIN_LOG" 2>/dev/null || echo 0)
  echo "$((now - start)) shm_mb=$shm workers=$workers worker_cpu_s=$worker_cpu load=$load steps=$steps" >> "$SAMPLE_LOG"
  sleep 2
done

echo "=== steps" >> "$SAMPLE_LOG"
grep "| samples" "$TRAIN_LOG" 2>/dev/null | sed -E 's/.*(\[step [0-9]+\]).*step ([0-9.]+)s.*data \(vae ([0-9.]+), wait ([0-9.]+)\).*/\1 step=\2 vae=\3 wait=\4/' >> "$SAMPLE_LOG"

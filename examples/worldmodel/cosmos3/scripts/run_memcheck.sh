#!/bin/bash
# Six steps of the current tree on one node: does the 8 GiB guard keep reserved memory flat, and does the AvgDown
# zero-pad skip hold the step time? Runs inside the nzf-cai-dev container.
set -uo pipefail

LOCAL=/mnt/disk0/jiangzs/nzf
BENCH=$LOCAL/bench_online
REPO=$LOCAL/ColossalAI

for _ in $(seq 1 240); do
  if grep -qa 'probe3 done' "$BENCH/logs/probe3.log" 2>/dev/null; then break; fi
  sleep 10
done
sleep 10

echo "=== memcheck $(date -u +%H:%M:%S)"
MASTER_PORT=29895 bash "$REPO/examples/worldmodel/cosmos3/scripts/bench_online_step.sh" memcheck 32 6 \
  --max-episodes 64 --action-cache "$LOCAL/action_cache" --worker-threads 4 --vae-content-pad 16 \
  > "$BENCH/logs/memcheck.log" 2>&1
echo "memcheck exit $? $(date -u +%H:%M:%S)"

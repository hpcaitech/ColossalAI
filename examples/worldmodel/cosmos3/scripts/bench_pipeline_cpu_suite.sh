#!/bin/bash
# The three cases that answer "what can the CPU half of the pipeline deliver on one 8-card node".
#
#   1. 4 workers, augmentation on   -- the production configuration
#   2. 8 workers, augmentation on   -- the untested lever (the node has 192 cores, 4 workers use 32 threads)
#   3. 4 workers, augmentation off  -- control: how much of the cost is the Compose itself
set -uo pipefail
LIMIT=${1:-640}
for case in "cpu_w4_aug 4 on" "cpu_w8_aug 8 on" "cpu_w4_noaug 4 off"; do
  set -- $case
  bash /home/niuzifan/scripts/bench_pipeline_cpu.sh "$1" "$2" "$3" "$LIMIT"
  echo
done

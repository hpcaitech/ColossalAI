#!/bin/bash
# Stop every benchmark job on this node and report what is left.
#
# Written as a file because the container wrapper joins its arguments into one `bash -c` string, so anything with
# quotes or a pipe has to live here rather than on the command line.
set -uo pipefail

for pattern in infrarsi train_sft.py precompute_latents diag_stall window_cost seek_cost sample_cost; do
  docker exec nzf-cai-dev pkill -9 -f "$pattern" 2>/dev/null || true
done
sleep 6

host=$(hostname -s)
train=$(docker exec nzf-cai-dev pgrep -cf train_sft.py 2>/dev/null || echo 0)
extract=$(docker exec nzf-cai-dev pgrep -cf precompute_latents 2>/dev/null || echo 0)
bench=$(docker exec nzf-cai-dev pgrep -cf "infrarsi|window_cost|sample_cost|seek_cost|diag_stall" 2>/dev/null || echo 0)
workers=$(docker exec nzf-cai-dev pgrep -cf spawn_main 2>/dev/null || echo 0)

# Devices report the baseline reservation when nobody holds them; a real holder shows up as a pid.
busy=0
for device in 0 1 2 3 4 5 6 7; do
  if npu-smi info -t proc-mem -i "$device" 2>/dev/null | grep -q "Process id"; then
    busy=$((busy + 1))
  fi
done

echo "$host train=$train extract=$extract bench=$bench workers=$workers npus_busy=$busy/8"

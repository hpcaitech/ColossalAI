#!/bin/bash
# One line per second: page-cache refill pressure and disk reads for the volume that holds /mnt/disk0.
# Runs on the *host* (the container shares the kernel counters, but this keeps it out of the training cgroup).
set -uo pipefail

OUT=${1:-/mnt/disk0/jiangzs/nzf/bench_online/io_sample.csv}
echo "t,pgmajfault,pgpgin,memavail_kb,dm0_read,nvme0_read,nvme1_read" > "$OUT"

while true; do
  maj=$(awk '/^pgmajfault/ {print $2}' /proc/vmstat)
  pgin=$(awk '/^pgpgin/ {print $2}' /proc/vmstat)
  avail=$(awk '/^MemAvailable/ {print $2}' /proc/meminfo)
  read -r dm0 nv0 nv1 < <(awk '$3 == "dm-0" {d = $6} $3 == "nvme0n1" {a = $6} $3 == "nvme1n1" {b = $6} END {print d, a, b}' /proc/diskstats)
  printf '%s,%s,%s,%s,%s,%s,%s\n' "$(date +%s.%N)" "$maj" "$pgin" "$avail" "$dm0" "$nv0" "$nv1" >> "$OUT"
  sleep 1
done

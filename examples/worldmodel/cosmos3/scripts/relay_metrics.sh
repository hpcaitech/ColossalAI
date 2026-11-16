#!/bin/bash
# Telemetry relay between the training nodes and the dashboard host, with atomic publication.
#
#   relay_metrics.sh out     # this node -> /home/niuzifan/nzf_sync/<tag>/<hostname>/   (on node7/8/9)
#   relay_metrics.sh in      # NFS staging -> <runs>/<tag>/metrics/                     (on the dashboard host)
#
# The dashboard host only has its own node's rank files locally; every other node's telemetry arrives here, and
# the run directory must look like one complete metrics tree to the monitor.
#
# **Everything is copied to a temp name and renamed into place.**  A plain ``cp -f`` truncates the target first,
# and a reader that stats during that window sees a short file: the monitor treats a shrinking JSONL as
# "rewritten from the start" and re-reads it, which duplicated per-rank series points and back-filled old steps
# into some ranks but not others (draft F39).  ``mv`` inside the same directory is atomic, so a reader sees
# either the old file or the new one, never a half-written one.  The temp names start with a dot, so the
# monitor's ``rank-*.jsonl`` / ``heartbeat-*.json`` / ``system/*.jsonl`` globs never pick them up.
#
# Runs every node-side push/pull on a 5 s tick; rank files (the big ones, ~2.4 MB per 2.5k steps) every 6th tick
# because heartbeats and system samples are what the dashboard needs promptly.
set -u

MODE=${1:?usage: relay_metrics.sh in|out}
HOST=$(cat /proc/sys/kernel/hostname)
RUNS=${RUNS:-/mnt/disk0/jiangzs/nzf/bench_online/runs}
STAGE=${STAGE:-/home/niuzifan/nzf_sync}
WINDOW_MIN=${WINDOW_MIN:-720}
RANK_EVERY=${RANK_EVERY:-6}

copy_atomic() {
  local src=$1 dst=$2 tmp
  tmp="$dst/.relay.$(basename "$src").$$"
  if cp -f "$src" "$tmp" 2>/dev/null; then
    mv -f "$tmp" "$dst/$(basename "$src")" 2>/dev/null || rm -f "$tmp"
  else
    rm -f "$tmp"
  fi
}

copy_glob() {
  local file
  for file in $1; do
    [ -e "$file" ] || continue
    copy_atomic "$file" "$2"
  done
}

copy_run() {
  local src=$1 dst=$2
  copy_glob "$src/system/*.jsonl" "$dst/system"
  copy_glob "$src/heartbeat-*.json" "$dst"
  if [ $((tick % RANK_EVERY)) -eq 0 ]; then
    copy_glob "$src/rank-*.jsonl" "$dst"
  fi
}

tick=0
while true; do
  if [ "$MODE" = "out" ]; then
    for run in $(find "$RUNS" -maxdepth 1 -mindepth 1 -type d -mmin -"$WINDOW_MIN" 2>/dev/null); do
      tag=$(basename "$run")
      src="$run/metrics"
      [ -d "$src" ] || continue
      dst="$STAGE/$tag/$HOST"
      mkdir -p "$dst/system" || continue
      copy_run "$src" "$dst"
    done
  else
    for stage_run in $(find "$STAGE" -maxdepth 1 -mindepth 1 -type d -mmin -"$WINDOW_MIN" 2>/dev/null); do
      tag=$(basename "$stage_run")
      dst="$RUNS/$tag/metrics"
      [ -d "$dst" ] || continue
      mkdir -p "$dst/system" || continue
      for dir in "$stage_run"/*/; do
        host=$(basename "$dir")
        [ "$host" = "$HOST" ] && continue
        copy_run "$dir" "$dst"
      done
    done
  fi
  tick=$((tick + 1))
  sleep 5
done

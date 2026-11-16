#!/bin/bash
# Pull every node's copy of a six-node run's per-rank metrics into one directory, for analyze_multinode_spread.py.
#
#   bash pull_6node_metrics.sh TAG [DEST]
#
# Each node keeps the run directory on its own local disk, so a single node's metrics/ only has its own 8 ranks.
# The files are written by the container as root, so they are streamed out with `docker exec tar` rather than rsynced.
set -euo pipefail

TAG=${1:?tag}
DEST=${2:-/tmp/6node_metrics}
NODES=${NODES:-"node1 node5 node6 node7 node8 node9"}
RUN=/mnt/disk0/jiangzs/nzf/bench_online/runs/$TAG

rm -rf "$DEST"
mkdir -p "$DEST"
index=0
for node in $NODES; do
  target=$DEST/node_rank_$index
  mkdir -p "$target"
  # Two hops on purpose: the ssh banner this cluster prints on every login would corrupt a streamed tar, so the
  # archive is written to the node's /tmp first and copied afterwards (scp is not affected by the banner).
  # tar exits 1 with "file changed as we read it" on a live run; that is expected, not a failure.
  ssh -o ConnectTimeout=15 "$node" "docker exec nzf-cai-dev tar cf - -C $RUN/metrics . > /tmp/$TAG.metrics.tar" >/dev/null || true
  scp -q "$node:/tmp/$TAG.metrics.tar" "$DEST/node_rank_$index.tar"
  tar xf "$DEST/node_rank_$index.tar" -C "$target"
  rm -f "$DEST/node_rank_$index.tar"
  index=$((index + 1))
done
echo "pulled $(ls "$DEST"/*/rank-*.jsonl 2>/dev/null | wc -l) rank files from ${index} nodes into $DEST"

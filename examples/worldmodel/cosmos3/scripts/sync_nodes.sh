#!/bin/bash
# Push this checkout's training/monitor code to the nodes, from the workstation.
#
#   bash scripts/sync_nodes.sh node1 node6
#
# The nodes run the code from /mnt/disk0/jiangzs/nzf/ColossalAI (node-local disk, not NFS: see
# docs/../draft_cosmos3_online_opt_zh.md F27).  Everything the trainer imports must go together --
# examples/worldmodel/cosmos3, .../cosmos3_harness and .../cosmos3_monitor -- plus colossalai/ for the
# plugin changes; syncing one directory alone produces an ImportError on the node (or, worse, a mixed
# pair of files that runs).
#
# No --delete: the node copy keeps files this checkout does not have, and a running job's logs and
# run directories live next to the source.  Bytecode caches are excluded because root-owned copies
# make rsync fail with EPERM.
set -uo pipefail

DEST=${DEST:-/mnt/disk0/jiangzs/nzf/ColossalAI}
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
TRIES=${TRIES:-6}

SYNC=(examples/worldmodel/cosmos3 examples/worldmodel/cosmos3_harness examples/worldmodel/cosmos3_monitor
      examples/__init__.py colossalai .pre-commit-config.yaml)

KEY_FILES="examples/worldmodel/cosmos3/train_sft.py examples/worldmodel/cosmos3_harness/checkpoint.py
           examples/worldmodel/cosmos3_monitor/server.py examples/worldmodel/cosmos3_monitor/static/app.js"

local_sums() {
  # shellcheck disable=SC2086
  (cd "$REPO" && md5 -q $KEY_FILES 2>/dev/null | cut -c1-8 | paste -sd, -)
}

push() {
  local node=$1 attempt=1
  while [ $attempt -le $TRIES ]; do
    # -R keeps examples/worldmodel/... under the destination; without it rsync flattens every source
    # to its basename and the node gets a second, stale copy next to the real tree.
    if (cd "$REPO" && rsync -aR --timeout=120 \
        --exclude '__pycache__' --exclude '*.pyc' --exclude '.pkl_memoize_py3' \
        --exclude 'runs' --exclude 'logs' --exclude '*.safetensors' \
        "${SYNC[@]}" "$node:$DEST/") 2>/tmp/sync_$node.err; then
      # md5s of the files that decide a run's behaviour, so a mixed checkout is visible at a glance
      local remote
      remote=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$node" "cd $DEST && md5sum $KEY_FILES 2>/dev/null | cut -c1-8 | paste -sd, -" 2>/dev/null | grep -E '^[0-9a-f,]+$' || true)
      echo "[sync] $node ok local=$(local_sums) remote=${remote:-<unavailable>}"
      return 0
    fi
    echo "[sync] $node attempt $attempt failed: $(tail -1 /tmp/sync_$node.err)" >&2
    attempt=$((attempt + 1))
    sleep 3
  done
  return 1
}

status=0
for node in "$@"; do
  push "$node" || status=1
done
exit $status

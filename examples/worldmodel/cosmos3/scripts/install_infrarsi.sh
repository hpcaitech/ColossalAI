#!/bin/bash
# Install InfraRSI into its own venv, deliberately separate from the training venv so that nothing here can
# disturb torch / torch_npu / diffusers.
#
# Written as a file rather than passed to dev-container-node.sh as one more argument: that wrapper forwards its
# arguments by joining them into a single `bash -c` string, so a nested `bash -c "..."` loses its quotes and the
# install either runs in the wrong shell or not at all.
set -uo pipefail

REPO=/mnt/disk0/jiangzs/nzf/InfraRSI
VENV=/mnt/disk0/jiangzs/nzf/infrarsi-venv
PY=/usr/local/python3.12.13/bin/python3

echo "== python"; "$PY" -V
if [ ! -x "$VENV/bin/python" ]; then
  echo "== creating venv"
  "$PY" -m venv "$VENV" || { echo "venv failed"; exit 1; }
fi

echo "== pip install -e ."
"$VENV/bin/pip" install -e "$REPO" 2>&1 | tail -n 12
echo "== pip exit ${PIPESTATUS[0]}"

echo "== version"
"$VENV/bin/infrarsi" --version 2>&1 | tail -n 3

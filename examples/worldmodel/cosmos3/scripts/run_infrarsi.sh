#!/bin/bash
# Run infrarsi inside the container, in the right working directory, with the LLM endpoint wired up.
#
# `dev-container-node.sh` starts every command in the container's own workdir and forwards its arguments by
# joining them into one `bash -c` string, so a `cd` on the host has no effect and quoted arguments lose their
# quotes. Doing the cd here, in a real script, is the only form that survives.
set -uo pipefail

PROJECT=/mnt/disk0/jiangzs/nzf/infrarsi-cosmos3
INFRARSI=/mnt/disk0/jiangzs/nzf/infrarsi-venv/bin/infrarsi

# The LLM is served on the workstation and reaches this node through a reverse ssh tunnel on loopback, so the
# key travels in the environment rather than in the manifest or the command line.
export MAGPIE_API_KEY=${MAGPIE_API_KEY:-magpie}
export INFRARSI_DEBUG=${INFRARSI_DEBUG:-0}

cd "$PROJECT" || exit 1
exec "$INFRARSI" "$@"

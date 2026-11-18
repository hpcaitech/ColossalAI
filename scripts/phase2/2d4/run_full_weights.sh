#!/usr/bin/env bash
# Host runner for both original-weight smoke tests, preserving either failure.
set -euo pipefail
cd /mnt/beegfs/ColossalAI/wangzhijian/src/ColossalAI
export D2D4_CONTAINER="${D2D4_CONTAINER:-wzj-colossalai-2d4-20261118}"
export D2D4_LOCK_TIMEOUT="${D2D4_LOCK_TIMEOUT:-600}"
export D2D4_GPUS="${D2D4_GPUS:-5,6}"
flock -w 600 "${D2D4_GPU_LOCK:-/tmp/colossalai-ci-gpu.lock}" true
status=0
for model in ouro14 ouro26; do
    if ! D2D4_TIMEOUT=900 bash scripts/phase2/2d4/run_host.sh full-weights --model "$model"; then
        status=1
    fi
done
exit "$status"

#!/usr/bin/env bash
set -euo pipefail
cd /mnt/beegfs/ColossalAI/wangzhijian/src/ColossalAI
export D2D4_CONTAINER="${D2D4_CONTAINER:-wzj-colossalai-2d4-20261118}"
export D2D4_LOCK_TIMEOUT="${D2D4_LOCK_TIMEOUT:-600}"
export D2D4_GPUS="${D2D4_GPUS:-5,6}"
flock -w 600 "${D2D4_GPU_LOCK:-/tmp/colossalai-ci-gpu.lock}" true
D2D4_OFFLINE_FIXTURES=1 bash scripts/phase2/2d4/run_host.sh tests \
    tests/test_booster/test_plugin/test_ouro.py -k 'lazy_checkpoint and ddp' -q

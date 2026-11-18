#!/usr/bin/env bash
# Host entry: serialize complete acceptance with any existing task/CI GPU user.
set -euo pipefail
cd /mnt/beegfs/ColossalAI/wangzhijian/src/ColossalAI
export D2D4_CONTAINER="${D2D4_CONTAINER:-wzj-colossalai-2d4-20261118}"
export D2D4_LOCK_TIMEOUT="${D2D4_LOCK_TIMEOUT:-600}"
export D2D4_GPUS="${D2D4_GPUS:-5,6}"
# Release this wait lock before run_host reacquires and checks occupancy itself.
flock -w 600 "${D2D4_GPU_LOCK:-/tmp/colossalai-ci-gpu.lock}" true
D2D4_OFFLINE_FIXTURES=1 bash scripts/phase2/2d4/run_host.sh tests \
    tests/test_booster/test_plugin/test_ouro.py \
    tests/test_checkpoint_io/test_gemini_offload_export.py \
    tests/test_checkpoint_io/test_gemini_master_resume.py -q --maxfail=3
D2D4_TIMEOUT=900 bash scripts/phase2/2d4/run_host.sh full-weights --model ouro14
D2D4_TIMEOUT=900 bash scripts/phase2/2d4/run_host.sh full-weights --model ouro26

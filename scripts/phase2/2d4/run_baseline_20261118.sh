#!/usr/bin/env bash
set -uo pipefail
root=/mnt/beegfs/ColossalAI/wangzhijian/src/ColossalAI
export D2D4_CONTAINER=wzj-colossalai-2d4-20261118 D2D4_GPUS=5,6 D2D4_TIMEOUT=360
status=0
for case in dp:gemini_bf16_default resume:zero2_bf16 resume:gemini_bf16; do
    scenario="${case%%:*}"
    plugin="${case#*:}"
    bash "$root/scripts/phase2/2d4/run_host.sh" --scenario "$scenario" --model ouro14 --loops 4 --plugin "$plugin" --world-size 2 --micro-batches 4
    rc=$?
    printf 'BASELINE %s exit=%s\n' "$case" "$rc"
    if [[ "$rc" -ne 0 ]]; then status="$rc"; fi
done
exit "$status"

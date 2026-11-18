#!/usr/bin/env bash
# Host launcher: cooperate with the CI GPU lock, check occupancy, then enter container.
set -euo pipefail
D2D4_HOST_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
D2D4_GPUS="${D2D4_GPUS:-0,1}"
exec 9>"${D2D4_GPU_LOCK:-/tmp/colossalai-ci-gpu.lock}"
if ! flock -w "${D2D4_LOCK_TIMEOUT:-30}" 9; then
    printf 'Timed out waiting for the shared GPU lock.\n' >&2
    exit 75
fi
IFS=, read -ra D2D4_GPU_LIST <<< "$D2D4_GPUS"
for D2D4_GPU in "${D2D4_GPU_LIST[@]}"; do
    nvidia-smi --id="$D2D4_GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits |
        awk -F, '{if ($1+0 > 256 || $2+0 > 0) exit 1}'
done
docker exec -e CUDA_VISIBLE_DEVICES="${D2D4_CUDA_VISIBLE_DEVICES:-0,1}" -e D2D4_TIMEOUT="${D2D4_TIMEOUT:-240}" \
    -e D2D4_OFFLINE_FIXTURES="${D2D4_OFFLINE_FIXTURES:-0}" -e D2D4_TEST_TIMEOUT="${D2D4_TEST_TIMEOUT:-1800}" \
    "${D2D4_CONTAINER:-wzj-colossalai-2d4}" bash -lc 'cd /workspace/ColossalAI && if [[ "${1:-}" == tests ]]; then shift; exec bash scripts/phase2/2d4/run_tests.sh "$@"; else exec bash scripts/phase2/2d4/run.sh "$@"; fi' bash "$@"

#!/usr/bin/env bash
# Run in the mounted development container; all outputs survive container exit.
set -euo pipefail
D2D4_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
D2D4_RESULTS="${D2D4_RESULTS:-/results/2d4}"
mkdir -p "$D2D4_RESULTS"
D2D4_RUN="$(mktemp -d "$D2D4_RESULTS/run-$(date -u +%Y%m%d-%H%M%S)-XXXXXX")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONOPTIMIZE=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MAX_JOBS=2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
printf 'Results: %s\n' "$D2D4_RUN"
{
    date -Is
    git -C "$D2D4_SCRIPT_DIR" status --short --branch
    git -C "$D2D4_SCRIPT_DIR" rev-parse HEAD
    sha256sum "$D2D4_SCRIPT_DIR"/*.py "$D2D4_SCRIPT_DIR"/*.sh
    if [[ "${1:-}" == "preflight" ]]; then
        timeout --signal=TERM --kill-after=15s 90s python -u "$D2D4_SCRIPT_DIR/../2d1/probe_cuda_device_access.py" --nccl
    elif [[ "${1:-}" == "full-weights" ]]; then
        shift
        timeout --signal=TERM --kill-after=30s "${D2D4_TIMEOUT:-900}s" python -u "$D2D4_SCRIPT_DIR/check_full_ouro.py" --out "$D2D4_RUN" "$@"
    else
        timeout --signal=TERM --kill-after=15s "${D2D4_TIMEOUT:-240}s" python -u "$D2D4_SCRIPT_DIR/check_ouro.py" --out "$D2D4_RUN" "$@"
    fi
} 2>&1 | tee "$D2D4_RUN/run.log"

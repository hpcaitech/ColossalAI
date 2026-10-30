#!/usr/bin/env bash
# Run inside the development container; preserve failure status through tee.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    printf 'Usage: bash %s {adapters|assets|imports|llada-tiny|llada-backward|remote-tiny|llada2-backward|ouro-backward|suite} [options]\n' "$0" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "$1" in
    adapters) SCRIPT_NAME=check_adapters.py ;;
    assets) SCRIPT_NAME=check_assets.py ;;
    suite) SCRIPT_NAME=run_suite.py ;;
    llada2-backward) SCRIPT_NAME=check_llada2_backward.py ;;
    imports) SCRIPT_NAME=check_imports.py ;;
    llada-tiny) SCRIPT_NAME=check_llada_tiny.py ;;
    llada-backward) SCRIPT_NAME=check_llada_backward.py ;;
    remote-tiny) SCRIPT_NAME=check_remote_tiny.py ;;
    ouro-backward) SCRIPT_NAME=check_ouro_backward.py ;;
    *)
        printf 'Unknown check: %s. Choose adapters, assets, imports, llada-tiny, llada-backward, remote-tiny, llada2-backward, ouro-backward, or suite.\n' "$1" >&2
        exit 2
        ;;
esac

LOG_ROOT="${D2D1_LOG_DIR:-/results/2d1}"
mkdir -p "$LOG_ROOT"
LOG_FILE="$(mktemp "$LOG_ROOT/$1-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")"
shift
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONOPTIMIZE=0
# These are CPU checks, so hide shared GPUs.
export CUDA_VISIBLE_DEVICES=""

printf 'Log: %s\n' "$LOG_FILE"
{
    printf 'Script: %s\n' "$SCRIPT_DIR/$SCRIPT_NAME"
    printf 'Started: %s\n' "$(date -Is)"
    printf 'Python: %s\n' "$(command -v python)"
    git -C "$SCRIPT_DIR" -c status.relativePaths=false status --short --branch
    git -C "$SCRIPT_DIR" rev-parse HEAD
    sha256sum "$SCRIPT_DIR/run.sh" "$SCRIPT_DIR"/*.py
    python -u "$SCRIPT_DIR/$SCRIPT_NAME" "$@"
} 2>&1 | tee "$LOG_FILE"

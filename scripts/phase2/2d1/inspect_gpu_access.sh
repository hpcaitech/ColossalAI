#!/usr/bin/env bash
# Run on the SSH host. Read-only comparison; never restarts a container or changes GPU state.
set -uo pipefail
LOG_ROOT="${D2D1_LOG_DIR:-/mnt/beegfs/ColossalAI/wangzhijian/containers/dev-torch213-tf517-peft0212/results/2d1}"
mkdir -p "$LOG_ROOT"
LOG_FILE="$(mktemp "$LOG_ROOT/gpu-access-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")"
{
    date -Is
    sha256sum "${BASH_SOURCE[0]}"
    printf 'HOST GPU INVENTORY\n'
    timeout 15 nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    host_rc=$?
    printf 'HOST_NVIDIA_SMI_EXIT=%s\n' "$host_rc"
    printf 'CONTAINER GPU INVENTORY\n'
    timeout 15 docker exec wzj-colossalai-dev-tf517 nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    container_rc=$?
    printf 'CONTAINER_NVIDIA_SMI_EXIT=%s\n' "$container_rc"
    docker inspect --format 'Memory={{.HostConfig.Memory}} NanoCpus={{.HostConfig.NanoCpus}} Devices={{json .HostConfig.DeviceRequests}}' wzj-colossalai-dev-tf517
    printf 'HOST MEMORY\n'
    free -h
    printf 'Log: %s\n' "$LOG_FILE"
    if (( host_rc != 0 || container_rc != 0 )); then exit 1; fi
} 2>&1 | tee "$LOG_FILE"
exit "${PIPESTATUS[0]}"

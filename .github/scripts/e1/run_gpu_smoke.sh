#!/usr/bin/env bash
# --check and --cpu-control never expose GPUs. --run requires two explicit UUIDs.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage:
  bash run_gpu_smoke.sh --check
  bash run_gpu_smoke.sh --cpu-control
  bash run_gpu_smoke.sh --run GPU-uuid-1 GPU-uuid-2

Default: --check. Obtain permission for both GPUs before using --run.
This script checks occupancy; it does not create or verify a reservation.
HELP
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
mode="${1:---check}"
case "$mode" in
    --help|-h) usage; exit 0 ;;
    --check|--cpu-control) [ "$#" -le 1 ] || die 'Unexpected arguments' ;;
    --run) [ "$#" -eq 3 ] || die '--run requires exactly two GPU UUIDs' ;;
    *) usage >&2; exit 2 ;;
esac

[ "$(hostname -s)" = gpu-h20-5 ] || die 'This initial configuration targets gpu-h20-5'
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image='nvcr.io/nvidia/pytorch@sha256:025d9b102b5436d4af8af58f12c6a46b7e5d16f19543b1d2cc4446bf2650b4f1'
for tool in docker timeout nvidia-smi sha256sum; do
    command -v "$tool" >/dev/null || die "Missing command: $tool"
done
docker image inspect "$image" >/dev/null || die 'The pinned image must already exist locally; no automatic pull'

gpu_args=(--runtime runc --env NVIDIA_VISIBLE_DEVICES=void)
python_args=(python /e1/gpu_smoke.py --check)
if [ "$mode" = --cpu-control ] || [ "$mode" = --run ]; then
    # A static loopback endpoint avoids hostname discovery in --network none.
    # Every run has a separate network namespace, so this port is not shared.
    python_args=(python -m torch.distributed.run --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 --max-restarts=0 /e1/gpu_smoke.py)
    [ "$mode" != --cpu-control ] || python_args+=(--cpu-control)
fi

if [ "$mode" = --run ]; then
    command -v flock >/dev/null || die 'Missing flock'
    [ "$2" != "$3" ] || die 'GPU UUIDs must be different'
    for gpu in "$2" "$3"; do
        [[ "$gpu" =~ ^GPU-[[:xdigit:]]{8}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{12}$ ]] || die 'Use full GPU UUIDs, not indices or all'
    done

    # Node-local advisory lock: coordinates this wrapper only, not other users' jobs.
    exec 9>"/tmp/colossalai-e1-ricardoo-${UID}.lock"
    flock -n 9 || die 'Another E1 GPU smoke test holds the node-local lock'

    for gpu in "$2" "$3"; do
        inventory="$(nvidia-smi -i "$gpu" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)"
        IFS=, read -r found memory utilization <<< "$inventory"
        memory="${memory//[[:space:]]/}"
        utilization="${utilization//[[:space:]]/}"
        [ "$found" = "$gpu" ] || die "GPU UUID mismatch: $gpu"
        [[ "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || die "Cannot determine occupancy: $gpu"
        [ "$memory" -le 256 ] && [ "$utilization" -eq 0 ] || die "GPU appears busy: $inventory"
        processes="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"
        [ -z "${processes//[[:space:]]/}" ] || die "GPU has a compute process: $gpu"
    done
    gpu_args=(--runtime nvidia --gpus "\"device=$2,$3\"" --env "NVIDIA_VISIBLE_DEVICES=$2,$3")
fi

results_root='/mnt/beegfs/ricardoo/ci/gpu-h20-5/test-results'
mkdir -p "$results_root"
result_dir="$(mktemp -d "$results_root/e1-$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")"
container_name="e1-smoke-$(basename "$result_dir")"
cid_file="$result_dir/container.cid"

cleanup() {
    local rc=$?
    trap - EXIT
    if [ -s "$cid_file" ]; then
        local cid
        cid="$(cat "$cid_file")"
        if [[ "$cid" =~ ^[[:xdigit:]]{64}$ ]]; then
            timeout 20s docker rm -f "$cid" >/dev/null 2>&1 || true
        fi
    fi
    printf '%s\n' "$rc" > "$result_dir/exit-code.txt"
    printf 'Mode: %s; exit code: %s; results: %s\n' "$mode" "$rc" "$result_dir"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

{
    printf 'utc=%s\nhost=%s\nmode=%s\nimage=%s\n' "$(date -u +%FT%TZ)" "$(hostname -s)" "$mode" "$image"
    if [ "$mode" = --run ]; then
        printf 'assigned_gpu_uuids=%s,%s\n' "$2" "$3"
    fi
    if git -C "$script_dir" rev-parse HEAD >/dev/null 2>&1; then
        printf 'checkout_sha=%s\n' "$(git -C "$script_dir" rev-parse HEAD)"
    fi
    sha256sum "$script_dir/gpu_smoke.py" "$script_dir/run_gpu_smoke.sh"
    docker image inspect "$image" --format 'image_id={{.Id}}'
    nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu --format=csv
} | tee "$result_dir/environment.txt"

# The in-container deadline also bounds execution if the SSH client disappears.
# A second host deadline plus EXIT cleanup handles a stuck docker client.
timeout --signal=TERM --kill-after=20s 300s \
    docker run --rm --init --pull never \
    --name "$container_name" --cidfile "$cid_file" \
    --network none --cpus 4 --memory 8g --memory-swap 8g --pids-limit 256 \
    --shm-size 1g --ulimit memlock=-1 \
    --cap-drop ALL --security-opt no-new-privileges \
    --user "$(id -u):$(id -g)" --workdir /tmp \
    --mount "type=bind,src=$script_dir,dst=/e1,readonly" \
    --env HOME=/tmp --env USER=colossalai-ci --env LOGNAME=colossalai-ci \
    --env TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor --env TRITON_CACHE_DIR=/tmp/triton \
    --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=1 \
    --env NCCL_DEBUG=WARN --env NCCL_IB_DISABLE=1 --env NCCL_SOCKET_IFNAME=lo \
    --env GLOO_SOCKET_IFNAME=lo --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    "${gpu_args[@]}" --entrypoint timeout "$image" \
    --signal=TERM --kill-after=15s 240s "${python_args[@]}" \
    2>&1 | tee "$result_dir/container.log"

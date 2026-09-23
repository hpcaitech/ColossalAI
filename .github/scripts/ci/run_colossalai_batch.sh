#!/usr/bin/env bash

set -Euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${COLOSSAL_REPO:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
RESULTS_ROOT="${COLOSSAL_RESULTS:-${RUNNER_TEMP:-/tmp}/colossalai-ci-results}"
BATCH="${1:-list}"
GPU_LIST="${2:-}"
FAST_MODE="${FAST_MODE:-1}"
MAXFAIL="${MAXFAIL:-10}"

BATCH="$(printf '%s' "${BATCH}" | tr '[:upper:]' '[:lower:]')"

print_manifest() {
    cat <<'EOF'
Usage:
  run_colossalai_batch.sh <batch> [physical_gpu_list]

Batches:
  cpu  CPU-only configuration, pipeline and tensor logic (no visible GPU)
  2    Single-GPU accelerator, FP16, LazyTensor and inference structures
  3    Single-GPU Adam/CPUAdam/HybridAdam compiled kernels
  4a   Single-GPU FP8 cast/linear/hook
  4b   Single-GPU inference CUDA kernels
  4c   Single-GPU Triton inference kernels
  4d   Single-GPU SmoothQuant tests
  4e   Single-GPU flash-decoding-attention parameter matrix
  5    Two-GPU DDP/FSDP/LowLevelZeRO/LoRA
  6    Four-GPU DeviceMesh/DTensor/communication foundations
  7    Four-GPU FP8 collectives and DDP/FSDP communication hooks
  8    Four-GPU Booster/Gemini/checkpoint/pipeline/ZeRO/MoE/optimizers
  9    Four-GPU non-largedist ShardFormer/inference/model tests
  10   Eight-GPU tests marked largedist
  11   Two-GPU Apex-dependent Gemini regression files

Environment:
  COLOSSAL_REPO                 source tree to test
  COLOSSAL_VENV                 Python virtual environment (optional)
  COLOSSAL_PYTHON               Python executable (optional)
  COLOSSAL_RESULTS              result directory
  COLOSSAL_EXPECT_TORCH_PREFIX  required torch version prefix (optional)
  COLOSSAL_CUDA_HOME            CUDA toolkit root (optional)
  COLOSSAL_CACHE_ROOT           local root for per-batch compiler caches (optional)
  FAST_MODE=0                   run the full model matrix
  TIMEOUT_MIN                   override the per-batch timeout
  MAXFAIL                       override pytest --maxfail (default: 10)
EOF
}

if [[ "${BATCH}" == "list" || "${BATCH}" == "help" || "${BATCH}" == "-h" || "${BATCH}" == "--help" ]]; then
    print_manifest
    exit 0
fi

declare -a TESTS=()
declare -a PYTEST_EXTRA=()
EXPECTED_GPUS=0
DEFAULT_TIMEOUT_MIN=60
LABEL=""
DESCRIPTION=""
NEEDS_CUDA_TOOLKIT=0
RUN_DDP_HOOK_SUPPLEMENT=0

case "${BATCH}" in
    cpu|1|01)
        LABEL="01-cpu-core"
        DESCRIPTION="CPU-only configuration, pipeline and tensor logic"
        EXPECTED_GPUS=0
        DEFAULT_TIMEOUT_MIN=15
        TESTS=(
            tests/test_config/test_load_config.py
            tests/test_infer/test_async_engine/test_request_tracer.py
            tests/test_optimizer/test_lr_scheduler.py
            tests/test_pipeline/test_pipeline_utils/test_t5_pipeline_utils.py
            tests/test_pipeline/test_pipeline_utils/test_whisper_pipeline_utils.py
            tests/test_pipeline/test_schedule/test_pipeline_schedule_utils.py
            tests/test_tensor/test_dtensor/test_dtensor_sharding_spec.py
            tests/test_tensor/test_shape_consistency.py
            tests/test_tensor/test_sharding_spec.py
        )
        ;;
    2|02)
        LABEL="02-single-gpu-core"
        DESCRIPTION="Accelerator, native FP16, LazyTensor and inference data structures"
        EXPECTED_GPUS=1
        DEFAULT_TIMEOUT_MIN=40
        TESTS=(
            tests/test_booster/test_accelerator.py
            tests/test_booster/test_mixed_precision/test_fp16_torch.py
            tests/test_lazy/test_ops.py
            tests/test_lazy/test_models.py
            tests/test_infer/test_batch_bucket.py
            tests/test_infer/test_config_and_struct.py
            tests/test_infer/test_kvcache_manager.py
            tests/test_infer/test_request_handler.py
        )
        ;;
    3|03)
        LABEL="03-adam-kernels"
        DESCRIPTION="FusedAdam, CPUAdam and HybridAdam compiled kernels"
        EXPECTED_GPUS=1
        DEFAULT_TIMEOUT_MIN=90
        NEEDS_CUDA_TOOLKIT=1
        TESTS=(
            tests/test_optimizer/test_adam_kernel.py
            tests/test_optimizer/test_adam_optim.py
        )
        ;;
    4a|04a)
        LABEL="04a-fp8-single-gpu"
        DESCRIPTION="FP8 cast, linear and parameter hook"
        EXPECTED_GPUS=1
        DEFAULT_TIMEOUT_MIN=30
        TESTS=(
            tests/test_fp8/test_fp8_cast.py
            tests/test_fp8/test_fp8_hook.py
            tests/test_fp8/test_fp8_linear.py
        )
        ;;
    4b|04b)
        LABEL="04b-inference-cuda-kernels"
        DESCRIPTION="InferenceOps CUDA kernels except flash decoding attention"
        EXPECTED_GPUS=1
        DEFAULT_TIMEOUT_MIN=90
        NEEDS_CUDA_TOOLKIT=1
        TESTS=(
            tests/test_infer/test_kernels/cuda/test_convert_fp8.py
            tests/test_infer/test_kernels/cuda/test_get_cos_and_sin.py
            tests/test_infer/test_kernels/cuda/test_kv_cache_memcpy.py
            tests/test_infer/test_kernels/cuda/test_rms_layernorm.py
            tests/test_infer/test_kernels/cuda/test_rotary_embdding_unpad.py
            tests/test_infer/test_kernels/cuda/test_silu_and_mul.py
        )
        ;;
    4c|04c)
        LABEL="04c-triton-inference-kernels"
        DESCRIPTION="Triton context/decoding attention, cache, RMSNorm and rotary kernels"
        EXPECTED_GPUS=1
        DEFAULT_TIMEOUT_MIN=90
        NEEDS_CUDA_TOOLKIT=1
        TESTS=(tests/test_infer/test_kernels/triton)
        ;;
    4d|04d)
        LABEL="04d-smoothquant"
        DESCRIPTION="SmoothQuant INT8 linear, attention, MLP and rotary tests"
        EXPECTED_GPUS=1
        DEFAULT_TIMEOUT_MIN=30
        NEEDS_CUDA_TOOLKIT=1
        TESTS=(tests/test_smoothquant)
        ;;
    4e|04e)
        LABEL="04e-flash-decoding-attention"
        DESCRIPTION="Flash decoding attention full parameter matrix"
        EXPECTED_GPUS=1
        DEFAULT_TIMEOUT_MIN=90
        NEEDS_CUDA_TOOLKIT=1
        TESTS=(tests/test_infer/test_kernels/cuda/test_flash_decoding_attention.py)
        ;;
    5|05)
        LABEL="05-booster-lora-2gpu"
        DESCRIPTION="DDP, FSDP, LowLevelZeRO and LoRA"
        EXPECTED_GPUS=2
        DEFAULT_TIMEOUT_MIN=90
        NEEDS_CUDA_TOOLKIT=1
        TESTS=(
            tests/test_booster/test_plugin/test_dp_plugin_base.py
            tests/test_booster/test_plugin/test_torch_ddp_plugin.py
            tests/test_booster/test_plugin/test_torch_fsdp_plugin.py
            tests/test_booster/test_plugin/test_low_level_zero_plugin.py
            tests/test_lora/test_lora.py
        )
        ;;
    6|06)
        LABEL="06-distributed-tensor-4gpu"
        DESCRIPTION="DeviceMesh, process groups, DTensor and communication specifications"
        EXPECTED_GPUS=4
        DEFAULT_TIMEOUT_MIN=60
        TESTS=(
            tests/test_cluster/test_device_mesh_manager.py
            tests/test_cluster/test_process_group_mesh.py
            tests/test_device/test_alpha_beta.py
            tests/test_device/test_device_mesh.py
            tests/test_device/test_extract_alpha_beta.py
            tests/test_device/test_init_logical_pg.py
            tests/test_device/test_search_logical_device_mesh.py
            tests/test_tensor/test_comm_spec_apply.py
            tests/test_tensor/test_dtensor/test_comm_spec.py
            tests/test_tensor/test_dtensor/test_dtensor.py
            tests/test_tensor/test_dtensor/test_layout_converter.py
            tests/test_tensor/test_mix_gather.py
            tests/test_tensor/test_padded_tensor.py
            tests/test_tensor/test_shape_consistency_apply.py
        )
        ;;
    7|07)
        LABEL="07-fp8-collectives-4gpu"
        DESCRIPTION="FP8 collectives and DDP/FSDP communication hooks"
        EXPECTED_GPUS=4
        DEFAULT_TIMEOUT_MIN=45
        RUN_DDP_HOOK_SUPPLEMENT=1
        TESTS=(
            tests/test_fp8/test_all_to_all_single.py
            tests/test_fp8/test_fp8_all_to_all.py
            tests/test_fp8/test_fp8_all_to_all_single.py
            tests/test_fp8/test_fp8_allgather.py
            tests/test_fp8/test_fp8_allreduce.py
            tests/test_fp8/test_fp8_fsdp_comm_hook.py
            tests/test_fp8/test_fp8_reduce_scatter.py
        )
        ;;
    8|08)
        LABEL="08-distributed-training-4gpu"
        DESCRIPTION="Booster, Gemini, checkpoint, pipeline, ZeRO, MoE and distributed optimizers"
        EXPECTED_GPUS=4
        DEFAULT_TIMEOUT_MIN=240
        NEEDS_CUDA_TOOLKIT=1
        TESTS=(
            tests/test_booster
            tests/test_checkpoint_io
            tests/test_pipeline
            tests/test_optimizer
            tests/test_zero
            tests/test_moe
        )
        PYTEST_EXTRA=(
            -m "not largedist"
            --ignore=tests/test_booster/test_accelerator.py
            --ignore=tests/test_booster/test_mixed_precision/test_fp16_torch.py
            --ignore=tests/test_booster/test_plugin/test_dp_plugin_base.py
            --ignore=tests/test_booster/test_plugin/test_torch_ddp_plugin.py
            --ignore=tests/test_booster/test_plugin/test_torch_fsdp_plugin.py
            --ignore=tests/test_booster/test_plugin/test_low_level_zero_plugin.py
            --ignore=tests/test_optimizer/test_lr_scheduler.py
            --ignore=tests/test_optimizer/test_adam_kernel.py
            --ignore=tests/test_optimizer/test_adam_optim.py
            --ignore=tests/test_pipeline/test_pipeline_utils/test_t5_pipeline_utils.py
            --ignore=tests/test_pipeline/test_pipeline_utils/test_whisper_pipeline_utils.py
            --ignore=tests/test_pipeline/test_schedule/test_pipeline_schedule_utils.py
            --ignore=tests/test_zero/test_gemini/test_grad_accum.py
            --ignore=tests/test_zero/test_gemini/test_grad_clip.py
        )
        ;;
    9|09)
        LABEL="09-model-and-inference-4gpu"
        DESCRIPTION="Non-largedist ShardFormer, inference engine and external-model tests"
        EXPECTED_GPUS=4
        DEFAULT_TIMEOUT_MIN=300
        NEEDS_CUDA_TOOLKIT=1
        TESTS=(
            tests/test_infer
            tests/test_lazy
            tests/test_shardformer
        )
        PYTEST_EXTRA=(
            -m "not largedist"
            --ignore=tests/test_infer/test_async_engine/test_request_tracer.py
            --ignore=tests/test_infer/test_batch_bucket.py
            --ignore=tests/test_infer/test_config_and_struct.py
            --ignore=tests/test_infer/test_kvcache_manager.py
            --ignore=tests/test_infer/test_request_handler.py
            --ignore=tests/test_infer/test_kernels/cuda
            --ignore=tests/test_infer/test_kernels/triton
            --ignore=tests/test_lazy/test_models.py
            --ignore=tests/test_lazy/test_ops.py
        )
        ;;
    10)
        LABEL="10-largedist-8gpu"
        DESCRIPTION="All tests marked largedist"
        EXPECTED_GPUS=8
        DEFAULT_TIMEOUT_MIN=360
        NEEDS_CUDA_TOOLKIT=1
        TESTS=(tests/test_infer tests/test_shardformer)
        PYTEST_EXTRA=(-m largedist)
        ;;
    11)
        LABEL="11-apex-gemini-2gpu"
        DESCRIPTION="Apex-dependent Gemini gradient accumulation and clipping"
        EXPECTED_GPUS=2
        DEFAULT_TIMEOUT_MIN=150
        NEEDS_CUDA_TOOLKIT=1
        TESTS=(
            tests/test_zero/test_gemini/test_grad_accum.py
            tests/test_zero/test_gemini/test_grad_clip.py
        )
        ;;
    *)
        printf 'Unknown batch: %s\n' "${BATCH}" >&2
        print_manifest >&2
        exit 2
        ;;
esac

if [[ ! -d "${REPO}/tests" ]]; then
    printf 'Repository not found: %s\n' "${REPO}" >&2
    exit 2
fi

if [[ -n "${COLOSSAL_VENV:-}" ]]; then
    if [[ ! -x "${COLOSSAL_VENV}/bin/python" ]]; then
        printf 'Virtual environment not found: %s\n' "${COLOSSAL_VENV}" >&2
        exit 2
    fi
    # shellcheck disable=SC1091
    source "${COLOSSAL_VENV}/bin/activate"
fi

PYTHON="${COLOSSAL_PYTHON:-python}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    printf 'Python executable not found: %s\n' "${PYTHON}" >&2
    exit 2
fi

if [[ "${EXPECTED_GPUS}" -eq 0 ]]; then
    if [[ -n "${GPU_LIST}" ]]; then
        printf 'CPU batch does not accept a GPU list.\n' >&2
        exit 2
    fi
    export CUDA_VISIBLE_DEVICES=""
else
    if [[ -z "${GPU_LIST}" ]]; then
        printf 'Batch %s requires %s physical GPU number(s).\n' "${BATCH}" "${EXPECTED_GPUS}" >&2
        exit 2
    fi
    IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
    if [[ "${#GPU_ARRAY[@]}" -ne "${EXPECTED_GPUS}" ]]; then
        printf 'Batch %s requires exactly %s visible GPU(s), got: %s\n' "${BATCH}" "${EXPECTED_GPUS}" "${GPU_LIST}" >&2
        exit 2
    fi
    export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
fi

export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export MAX_JOBS="${MAX_JOBS:-4}"
COMPILE_CACHE_SCOPE="${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-${BATCH}-${BASHPID}"
COMPILE_CACHE_ROOT="${COLOSSAL_CACHE_ROOT:-${TMPDIR:-/tmp}/colossalai-ci-cache}/${COMPILE_CACHE_SCOPE}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${COMPILE_CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${COMPILE_CACHE_ROOT}/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${RUNNER_TEMP:-/tmp}/torch-extensions}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

if [[ "${FAST_MODE}" == "1" ]]; then
    export FAST_TEST=1
else
    unset FAST_TEST
fi

if [[ -n "${COLOSSAL_CUDA_HOME:-}" ]]; then
    export CUDA_HOME="${COLOSSAL_CUDA_HOME}"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi
if [[ "${NEEDS_CUDA_TOOLKIT}" == "1" && ( -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ) ]]; then
    printf 'Batch %s requires CUDA_HOME with an executable bin/nvcc.\n' "${BATCH}" >&2
    exit 2
fi

mkdir -p "${RESULTS_ROOT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"
cd "${REPO}"

EXPECTED_GPUS="${EXPECTED_GPUS}" COLOSSAL_EXPECT_TORCH_PREFIX="${COLOSSAL_EXPECT_TORCH_PREFIX:-}" "${PYTHON}" - <<'PY'
import os
import sys

import torch

expected = int(os.environ["EXPECTED_GPUS"])
prefix = os.environ.get("COLOSSAL_EXPECT_TORCH_PREFIX")
print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("visible GPU count:", torch.cuda.device_count())
if prefix:
    assert torch.__version__.startswith(prefix), (torch.__version__, prefix)
if expected == 0:
    assert not torch.cuda.is_available(), "CPU batch unexpectedly has CUDA access"
    assert torch.cuda.device_count() == 0
else:
    assert torch.cuda.is_available(), "CUDA is not available"
    assert torch.cuda.device_count() == expected, (torch.cuda.device_count(), expected)
    for index in range(expected):
        print(f"cuda:{index}:", torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index))
PY

if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    printf 'CUDA_HOME=%s\n' "${CUDA_HOME}"
    "${CUDA_HOME}/bin/nvcc" --version | tail -4
fi

TIMEOUT_MIN="${TIMEOUT_MIN:-${DEFAULT_TIMEOUT_MIN}}"
RUN_ID="${GITHUB_RUN_ID:-local}-$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_PREFIX="${RESULTS_ROOT}/${RUN_ID}-${LABEL}"

printf 'batch=%s\n' "${BATCH}"
printf 'description=%s\n' "${DESCRIPTION}"
printf 'physical_gpus=%s\n' "${GPU_LIST:-none}"
printf 'FAST_TEST=%s\n' "${FAST_TEST-unset}"
printf 'timeout_minutes=%s\n' "${TIMEOUT_MIN}"
printf 'result_prefix=%s\n' "${RESULT_PREFIX}"

run_pytest_batch() {
    set +e
    timeout --signal=TERM --kill-after=60s "${TIMEOUT_MIN}m" \
        "${PYTHON}" -m pytest \
        -v -s -ra --tb=short \
        --maxfail="${MAXFAIL}" \
        --durations=20 \
        --junitxml="${RESULT_PREFIX}.xml" \
        "${PYTEST_EXTRA[@]}" \
        "${TESTS[@]}" \
        2>&1 | tee "${RESULT_PREFIX}.log"
    local pytest_status="${PIPESTATUS[0]}"
    set -e
    return "${pytest_status}"
}

if run_pytest_batch; then
    MAIN_STATUS=0
else
    MAIN_STATUS=$?
fi

SUPPLEMENT_STATUS=0
if [[ "${RUN_DDP_HOOK_SUPPLEMENT}" == "1" ]]; then
    DDP_HOOK_FILE="tests/test_fp8/test_fp8_ddp_comm_hook.py"
    set +e
    COLLECT_OUTPUT="$("${PYTHON}" -m pytest --collect-only -q "${DDP_HOOK_FILE}" 2>&1)"
    COLLECT_STATUS=$?
    set -e
    if [[ "${COLLECT_STATUS}" -eq 0 && "${COLLECT_OUTPUT}" == *"::test_"* ]]; then
        set +e
        timeout --signal=TERM --kill-after=60s "${TIMEOUT_MIN}m" \
            "${PYTHON}" -m pytest -v -s -ra --tb=short --maxfail="${MAXFAIL}" \
            --junitxml="${RESULT_PREFIX}-ddp-hook.xml" "${DDP_HOOK_FILE}" \
            2>&1 | tee "${RESULT_PREFIX}-ddp-hook.log"
        SUPPLEMENT_STATUS="${PIPESTATUS[0]}"
        set -e
    else
        set +e
        timeout --signal=TERM --kill-after=60s "${TIMEOUT_MIN}m" \
            "${PYTHON}" "${DDP_HOOK_FILE}" 2>&1 | tee "${RESULT_PREFIX}-ddp-hook.log"
        SUPPLEMENT_STATUS="${PIPESTATUS[0]}"
        set -e
    fi
fi

printf 'main_exit_code=%s\n' "${MAIN_STATUS}"
printf 'supplement_exit_code=%s\n' "${SUPPLEMENT_STATUS}"
if [[ "${MAIN_STATUS}" -eq 0 && "${SUPPLEMENT_STATUS}" -eq 0 ]]; then
    printf '%s_exit_code=0\n' "${LABEL}"
    exit 0
fi
printf '%s_exit_code=1\n' "${LABEL}"
exit 1

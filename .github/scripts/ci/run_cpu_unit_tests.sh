#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${COLOSSAL_REPO:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
RESULTS_ROOT="${COLOSSAL_RESULTS:-${RUNNER_TEMP:-/tmp}/colossalai-fork-pr-results}"
PYTHON="${COLOSSAL_PYTHON:-python}"

TESTS=(
    .github/workflows/scripts/example_checks/test_run_test_ci.py
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

if [[ ! -d "${REPO}/tests" ]]; then
    printf 'Repository not found: %s\n' "${REPO}" >&2
    exit 2
fi

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    printf 'Python executable not found: %s\n' "${PYTHON}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES=""
export FAST_TEST="${FAST_MODE:-1}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${RESULTS_ROOT}"
cd "${REPO}"

"${PYTHON}" - <<'PY'
import os
import sys

import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("visible GPU count:", torch.cuda.device_count())
assert not torch.cuda.is_available(), "CPU gate unexpectedly has CUDA access"
assert torch.cuda.device_count() == 0
PY

RESULT_PREFIX="${RESULTS_ROOT}/cpu-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}"
set +e
timeout --signal=TERM --kill-after=60s 15m \
    "${PYTHON}" -m pytest \
    -v -ra --tb=short \
    --maxfail=10 \
    --durations=20 \
    --junitxml="${RESULT_PREFIX}.xml" \
    "${TESTS[@]}" \
    2>&1 | tee "${RESULT_PREFIX}.log"
PYTEST_STATUS="${PIPESTATUS[0]}"
set -e

printf 'pytest_exit_code=%s\n' "${PYTEST_STATUS}"
exit "${PYTEST_STATUS}"

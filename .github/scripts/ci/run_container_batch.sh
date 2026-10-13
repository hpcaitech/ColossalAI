#!/usr/bin/env bash
# Runs inside the dependency image, as the runner's non-root UID/GID.
set -euo pipefail

BATCH="${1:?batch is required}"
GPU_LIST="${2:?container-local GPU list is required}"
# A changed recipe/lock must not accidentally pass CI using an older image.
/opt/venv/bin/python /ci-source/.github/scripts/ci/container_batch.py \
    --check-recipe /ci-source/docker/torch213 --image-inputs /opt/ci
WORK_ROOT="$(mktemp -d /tmp/colossalai-batch.XXXXXX)"
export HOME="${WORK_ROOT}/home"
export USER=colossalai-ci
export LOGNAME=colossalai-ci
export TMPDIR="${WORK_ROOT}/tmp"
export RUNNER_TEMP="${WORK_ROOT}/cache"
export COLOSSAL_CACHE_ROOT="${RUNNER_TEMP}"
export TORCH_EXTENSIONS_DIR="${RUNNER_TEMP}/torch-extensions"
export COLOSSAL_RESULTS=/ci-results
export COLOSSAL_REPO="${WORK_ROOT}/repo"
export COLOSSAL_CUDA_HOME=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda
export COLOSSAL_EXPECT_TORCH_PREFIX=2.13.
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
unset COLOSSAL_VENV PYTHONPATH PYTEST_ADDOPTS
mkdir -p "${HOME}" "${TMPDIR}" "${RUNNER_TEMP}" "${COLOSSAL_REPO}"

# The read-only bind is only an input. Builds and tests can mutate this private
# copy without leaving root-owned artifacts or stale .so files in the checkout.
cp -R --no-preserve=ownership /ci-source/. "${COLOSSAL_REPO}/"
chmod -R u+w "${COLOSSAL_REPO}"

# A writable per-container venv reuses the image's read-only dependencies via a
# .pth file. No multi-GB Torch copy, network install or host venv is necessary.
IMAGE_PYTHON=/opt/venv/bin/python
"${IMAGE_PYTHON}" -m venv --without-pip "${WORK_ROOT}/venv"
export COLOSSAL_PYTHON="${WORK_ROOT}/venv/bin/python"
export PATH="${WORK_ROOT}/venv/bin:/usr/local/cuda/bin:${PATH}"
IMAGE_SITE="$("${IMAGE_PYTHON}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
LOCAL_SITE="$("${COLOSSAL_PYTHON}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
printf '%s\n' "${IMAGE_SITE}" > "${LOCAL_SITE}/ci-image-dependencies.pth"

cd "${COLOSSAL_REPO}"
"${COLOSSAL_PYTHON}" - <<'PY'
import json
import os
import sys
from pathlib import Path

import torch

assert torch.__version__.startswith("2.13."), torch.__version__
assert torch.version.cuda == "13.0", torch.version.cuda
assert torch.cuda.is_available(), "CUDA is not available inside the container"
assert torch.cuda.device_count() == len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
profile = {
    "source_sha": os.environ["CI_SOURCE_SHA"],
    "image_id": os.environ["CI_IMAGE_ID"],
    "python": sys.version,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}
print(json.dumps(profile, indent=2), flush=True)
Path("/ci-results/runtime.json").write_text(json.dumps(profile, indent=2) + "\n")
PY

# Batch 3 explicitly validates the AOT build. Other batches compile the exact
# source's kernels on demand (JIT) in disposable, batch-local caches.
BUILD_EXT=0
if [[ "${BATCH}" == "3" ]]; then
    BUILD_EXT=1
fi
export BUILD_EXT
printf 'Installing source %s with BUILD_EXT=%s\n' "${CI_SOURCE_SHA}" "${BUILD_EXT}"
"${COLOSSAL_PYTHON}" -m pip install --no-deps --no-build-isolation -e .
"${COLOSSAL_PYTHON}" -m pip check
exec bash .github/scripts/ci/run_colossalai_batch.sh "${BATCH}" "${GPU_LIST}"

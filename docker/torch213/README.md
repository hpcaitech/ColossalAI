# Torch 2.13 / CUDA 13.0 CI dependency image

This image provides an isolated Python environment and CUDA build tools for the
existing layered GPU tests. It contains **no ColossalAI checkout, installed
ColossalAI distribution or prebuilt ColossalAI extensions**. Install the exact
CI source commit after starting a container. This prevents a new PR's Python
code from using an older image's CUDA kernels.

The image recipe does not change runner selection, GPU allocation, required
checks, the release image or `version.txt`. Switching the reusable GPU workflow
from its host virtual environment to containers is a separate change.

## Dependency profile

| Component | Pinned profile |
| --- | --- |
| Platform | Linux x86_64, Ubuntu 22.04, Python 3.10 |
| CUDA Toolkit | 13.0.3 (nvcc reports 13.0.88) |
| PyTorch | 2.13.0+cu130 |
| torchvision | 0.28.0+cu130 |
| torchaudio | 2.11.0+cu130 (existing development artifact; see below) |
| Triton | 3.7.1 |
| Transformers / PEFT | 4.51.3 / 0.13.2 |
| pytest / pytest-testmon | 7.4.4 / 2.2.0 |
| setuptools | 84.0.0 |
| PyTorch's Python cuDNN package | nvidia-cudnn-cu13==9.20.0.48 |

The base image also contains a system cuDNN library (9.14 in the original image).
It is distinct from the Python wheel used by PyTorch. Use
`torch.backends.cudnn.version()` to inspect the PyTorch runtime; the base image's
labels alone do not describe that runtime.

`requirements-torch.lock` pins the four core wheel artifacts, including SHA256
hashes. `requirements.lock` pins the rest of the existing development environment,
including transitive dependencies and packaging tools. All entries are installed
with `--no-deps`, then `pip check` verifies that the snapshot is complete. This is
a Python 3.10/Linux profile, not a universal lock for all supported platforms.
Non-core packages are version-pinned but not hash-locked; Ubuntu packages still
come from apt repositories. For published images record the resolved base digest
and publish/use the final image by digest.

The existing torchaudio wheel has no `Requires-Dist: torch` declaration. Its
successful `pip check` is therefore **not** evidence of general TorchAudio 2.13
compatibility. Image validation covers importing it and CPU resampling. Keep this
artifact pinned for reproducing the current environment; separately validate any
audio workload or replacement wheel before changing it.

Optional backends **Apex, flash-attn, vLLM, TensorNVMe and SmoothQuant extensions**
are not included. Tests that require them may skip. In particular, the Apex batch
requires the legacy `apex.amp` API, not merely an arbitrary Apex installation.
Report skipped coverage separately; this image is not a claim of passing every
optional test. The general `requirements/requirements-test.txt` still includes
flash-attn, so do not reinstall it blindly on top of this locked core profile.

The library permits `torch>=2.2.0,<2.14`, retaining the 2.5.1 CPU baseline. That
installation range is distinct from this exact GPU profile and does not imply
that every intermediate Torch/CUDA combination has been tested. Transformers
remains at 4.51.3. On Torch 2.13 the FP8 implementation currently uses its eager
fallback; passing FP8 tests does not establish `torch.compile` support.

## Build

Use a dedicated or explicitly approved build host. A normal Docker build/pull
writes to the shared daemon's storage outside the checkout; keeping this recipe
in a personal directory does **not** confine those writes to that directory.
Do not run these commands on a shared server when only personal-directory writes
are permitted. The runtime example below also requires explicit GPU allocation.

Run from the repository root. Docker with BuildKit is required; GPUs are not
needed during the dependency image build. The Dockerfile-specific ignore file
restricts the context to the lock files, verification script and optional wheels.
The default package mirror is Huawei Cloud; core wheels use the PyTorch cu130
index. An approved registry mirror or base digest can be supplied via `CUDA_IMAGE`.

```bash
# Build a candidate from the committed recipe; use a fresh tag for each revision.
docker build --progress=plain \
  --build-arg SOURCE_COMMIT="$(git rev-parse HEAD)" \
  -f docker/torch213/Dockerfile \
  -t colossalai-ci:torch2.13-cu130-candidate .

# For a mirror/pinned base, add these build arguments to the command above.
# --build-arg CUDA_IMAGE=your-approved-registry/cuda@sha256:BASE_DIGEST
# --build-arg PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple
```

If core downloads are slow, place the four matching wheels in
`docker/torch213/wheelhouse/` first. They are git-ignored and mounted only during
the build; they are not copied into the final image. A nonempty wheelhouse must
contain **all four** pinned wheels. Pip verifies the hashes and fails on missing
or different artifacts rather than silently downloading replacements. Otherwise
the builder downloads the pinned artifacts from `TORCH_INDEX_URL`.

```bash
# Optional: download the pinned core artifacts before building.
python -m pip download --no-deps --require-hashes \
  --only-binary=:all: --platform manylinux_2_28_x86_64 \
  --platform manylinux_2_27_x86_64 --python-version 310 --implementation cp --abi cp310 \
  --index-url https://download.pytorch.org/whl/cu130 \
  -r docker/torch213/requirements-torch.lock \
  --dest docker/torch213/wheelhouse

# Verify dependencies and CPU extension operations without exposing any GPUs.
# Explicit runc avoids inheriting a host's default NVIDIA runtime.
docker run --rm --runtime=runc \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  colossalai-ci:torch2.13-cu130-candidate \
  python /opt/ci/verify_image.py
```

## Validate a source commit in the container

First obtain an authorized idle GPU. The following example uses physical GPU 4;
replace it with the allocated GPU or UUID. The container sees that device as
logical `cuda:0`. The host needs a compatible NVIDIA driver, Docker and NVIDIA
Container Toolkit; it does not need the personal Python venv or CUDA toolkit.

```bash
# Use a clean checkout of the exact revision to test, not a shared development tree.
CI_SOURCE_DIR="$PWD"
CI_RESULTS_DIR="$PWD/test-results/torch213-container"
mkdir -p "$CI_RESULTS_DIR"

# Compile this checkout's extensions after the allocated GPU is visible.
# --no-deps keeps the image pins; pip check still validates package metadata.
docker run --rm \
  --gpus '"device=4"' --shm-size=8g \
  --mount "type=bind,src=$CI_SOURCE_DIR,dst=/workspace" \
  --mount "type=bind,src=$CI_RESULTS_DIR,dst=/results" \
  -w /workspace \
  -e COLOSSAL_REPO=/workspace -e COLOSSAL_RESULTS=/results \
  -e COLOSSAL_VENV= -e COLOSSAL_PYTHON=/opt/venv/bin/python \
  -e COLOSSAL_CUDA_HOME=/usr/local/cuda -e COLOSSAL_EXPECT_TORCH_PREFIX=2.13. \
  -e TORCH_CUDA_ARCH_LIST=9.0 -e MAX_JOBS=4 \
  -e CUDA_VISIBLE_DEVICES=0 -e FAST_MODE=0 \
  colossalai-ci:torch2.13-cu130-candidate bash -euc '
    BUILD_EXT=1 python -m pip install --no-deps --no-build-isolation -e .
    python -m pip check
    python -c "import colossalai; print(colossalai.__file__)"
    bash .github/scripts/ci/run_colossalai_batch.sh 4a 0
  '
```

Batch 4a is the FP8 cast/linear/hook regression and does not reduce coverage based
on `FAST_TEST`. Batch 3 (Adam kernels) and 4b (inference CUDA kernels) are also
unaffected. The existing Layer 2 selects batches 2 and 5 with `FAST_MODE=1`, which
enables `FAST_TEST` in tests that reduce their model matrices. Use `FAST_MODE=0`
for Layer 3; the batch script **unsets** `FAST_TEST` instead of setting it to zero.

For CI, hold the existing host GPU lock from allocation through container exit,
mount only the exact source checkout/results, and stop/remove the named container
on cancellation before releasing the lock. Killing only the `docker run` client
does not provide that guarantee. Cache extensions by image digest, source revision
and GPU architecture; do not reuse an unrelated checkout's prebuilt `.so` files.

Build/validate the image on the actual runner and distribute it through the team
registry before switching the reusable workflow. A local image on another server
does not automatically exist in the runner's Docker daemon. This recipe neither
publishes an image nor changes the release Dockerfile or GitHub required checks.

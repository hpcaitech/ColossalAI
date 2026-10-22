# Torch 2.13 / CUDA 13.0 / Transformers 5.17 CI dependency image

This image provides an isolated Python environment and CUDA build tools for the
existing layered GPU tests. It contains **no ColossalAI checkout, installed
ColossalAI distribution or prebuilt ColossalAI extensions**. Install the exact
CI source commit after starting a container. This prevents a new PR's Python
code from using an older image's CUDA kernels.

The reusable GPU workflow uses this dependency-image profile while retaining
host-side GPU allocation and locking. Runner labels, required checks, the user
release image and `version.txt` are unchanged. The first CPU dependency check
and actual container/GPU qualification are separate acceptance steps.

## Dependency profile

| Component | Pinned profile |
| --- | --- |
| Platform | Linux x86_64, Ubuntu 22.04, Python 3.10 |
| CUDA Toolkit | 13.0.3 (nvcc reports 13.0.88) |
| PyTorch | 2.13.0+cu130 |
| torchvision | 0.28.0+cu130 |
| torchaudio | 2.11.0+cu130 (existing development artifact; see below) |
| Triton | 3.7.1 |
| Transformers / PEFT | 5.17.0 / 0.21.2 |
| Hugging Face Hub / Tokenizers | 1.5.0 / 0.23.1 |
| Diffusers | 0.35.2 |
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
and use the final local image by its full immutable image ID. A registry is optional.

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

The library permits `torch>=2.5,<2.14`, matching Transformers 5.17's minimum Torch
requirement and retaining the 2.5.1 CPU baseline. That
installation range is distinct from this exact GPU profile and does not imply
that every intermediate Torch/CUDA combination has been tested. Transformers and
PEFT are pinned to 5.17.0 and 0.21.2 respectively. The image verifier runs two
offline CPU optimizer steps on a tiny Llama with LoRA, checking finite losses,
finite nonzero gradients and actual adapter updates. It then saves and reloads
both `.bin` and safetensors adapters, checks adapter weights and evaluation
logits, and verifies another optimizer update after loading. These checks run
both without a process group and with a single-rank CPU Gloo group. PEFT 0.19.1
attempts to import the removed `EmbeddingParallel` symbol only when a process
group is initialized, so ordinary single-process training and reload checks
alone miss this regression. PEFT 0.21.2 includes the upstream TP compatibility
fixes; the image must be rebuilt when updating this pin.

This is a dependency smoke check, not a substitute for ColossalAI's
GPU/distributed regressions. On Torch
2.13 the FP8 implementation currently uses its eager
fallback; passing FP8 tests does not establish `torch.compile` support.

Diffusers is upgraded from 0.29.0 to 0.35.2 because the older version imports
`FLAX_WEIGHTS_NAME` from Transformers when resolving `DiffusionPipeline`; that
symbol is no longer available in Transformers 5.17. The image verifier also
resolves the DiffusionPipeline, PixArt Alpha and Stable Diffusion 3 lazy imports,
which `pip check` and a plain `import diffusers` do not exercise. These checks
do not establish numerical correctness of diffusion inference.

Do not switch the shared `COLOSSAL_CI_IMAGE` variable to a `docker commit`
experiment that still embeds the previous `/opt/ci` recipe. Build and qualify
a fresh candidate from these inputs first; source requirements, image locks and
embedded recipe must agree. Keep the stable image available for rollback, and
coordinate changing the shared image variable with merging the dependency PR.

## Small offline source checks

Inside a development container using this dependency profile, install the source
without replacing the locked packages, then run a CPU-only HF5 subset:

```bash
# Install the checkout, without compiling all optional CUDA extensions up front.
BUILD_EXT=0 python -m pip install --no-deps --no-build-isolation -e .
python -m pip check

# These selected tests do not use FAST_TEST; do not confuse them with full GPU CI.
unset FAST_TEST
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m pytest -q -ra \
    docker/torch213/test_verify_image.py \
    tests/test_lazy/test_chatglm_cpu.py \
    tests/test_shardformer/test_hf5_compat/test_cpu_rng.py \
    tests/test_shardformer/test_hf5_compat/test_integrated_core.py \
    tests/test_shardformer/test_hf5_compat/test_integration_alias.py \
    tests/test_shardformer/test_hf5_compat/test_integration_loading.py \
    tests/test_shardformer/test_hf5_compat/test_qwen_loss.py
```

This covers ChatGLM initialization/lazy materialization, LoRA checkpoint
roundtrips, HF5 policy dispatch, small non-TP/EP training, checkpoint/loading
contracts and single-rank CPU Gloo Qwen losses. The new ChatGLM and LoRA checks
also run in the Layer 1 and fork-safe CPU gates. This does not validate CUDA kernels,
tensor/expert parallelism, multi-GPU communication or the Torch 2.5.1 CPU baseline.
The complete image verifier must run **before** installing ColossalAI: it
deliberately rejects images containing an installed ColossalAI distribution.

## Build

Use a dedicated or explicitly approved build host. A normal Docker build/load/run
writes to the shared daemon's storage outside the checkout; keeping this recipe
in a personal directory does **not** confine those writes to that directory.
Do not run these commands on a shared server when only personal-directory writes
are permitted. The runtime example below also requires explicit GPU allocation.

Run from the repository root. Docker with BuildKit is required; GPUs are not
needed during the dependency image build. The Dockerfile-specific ignore file
restricts the context to the lock files, verification script and optional wheels.
The default package mirror is Huawei Cloud; core wheels use the PyTorch cu130
index. An approved registry mirror or base digest can be supplied via `CUDA_IMAGE`.
The recipe uses BuildKit's bundled Dockerfile frontend so it does not need an
extra `docker/dockerfile` download from Docker Hub. The inspected build host
uses BuildKit 0.27.1; use a recent BuildKit with support for bind/cache mounts.
GPU devices are hidden by default, and the CI executor overrides that setting
only with its allocated UUIDs.

```bash
# Build a candidate from the committed recipe; use a fresh tag for each revision.
docker build --platform linux/amd64 --progress=plain \
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

## Keep the image on the SSH server (no registry required)

Building with `docker build` already puts the image into that Docker daemon's
local image store. Record its full image ID and use that same daemon for tests
and the runner. A tag is convenient for humans but can be overwritten; CI pins
the content ID. No image upload, registry account or publishing workflow is
needed. The initial build still downloads base images and dependencies unless
they are already cached or supplied locally.

```bash
# Record the full immutable ID from the local Docker store; this does not pull.
export COLOSSAL_CI_IMAGE="$(docker image inspect --format '{{.Id}}' colossalai-ci:torch2.13-cu130-candidate)"
printf 'CI image ID: %s\n' "$COLOSSAL_CI_IMAGE"
```

Optionally save an image archive in an approved personal directory. Set
`CI_IMAGE_ARCHIVE_ROOT`, for example, to
`/mnt/beegfs/ColossalAI/wangzhijian/images`. Each export below gets a new directory
so it does not overwrite an existing archive. Do not commit large image tars to Git.

```bash
# Export a local backup and record both its checksum and Docker image ID.
: "${CI_IMAGE_ARCHIVE_ROOT:?Set an approved directory for image archives}"
: "${COLOSSAL_CI_IMAGE:?Set the full local sha256:image-ID}"
mkdir -p "$CI_IMAGE_ARCHIVE_ROOT"
CI_IMAGE_BUNDLE="$(mktemp -d "$CI_IMAGE_ARCHIVE_ROOT/torch213-cu130.XXXXXX")"
docker image save --output "$CI_IMAGE_BUNDLE/image.tar" "$COLOSSAL_CI_IMAGE"
printf '%s\n' "$COLOSSAL_CI_IMAGE" > "$CI_IMAGE_BUNDLE/image.id"
(cd "$CI_IMAGE_BUNDLE" && sha256sum image.tar > image.tar.sha256)
printf 'Saved image bundle: %s\n' "$CI_IMAGE_BUNDLE"

# Only when restoring or provisioning another daemon: check the archive first,
# then load it. Do not repeat this on every CI run if the image is already present.
(cd "$CI_IMAGE_BUNDLE" && sha256sum --check image.tar.sha256) && \
  docker image load --input "$CI_IMAGE_BUNDLE/image.tar"
export COLOSSAL_CI_IMAGE="$(tr -d '\n' < "$CI_IMAGE_BUNDLE/image.id")"
docker image inspect --format '{{.Id}}' "$COLOSSAL_CI_IMAGE"
```

The tar file and Docker's active storage are different things. Docker does not
run directly from `image.tar`: loading it writes layers/metadata to daemon
storage, and starting containers writes there too. A personal archive path does
not bypass shared-server restrictions. Do not change the shared daemon's data
root, restart it, prune unrelated data or claim this is zero-impact on the host.
If all writes must remain private, arrange an administrator-approved isolated
runtime first; this change does not install one automatically.

## Validate before opening the single PR

Use the unmerged branch's scripts directly; they are the same entry points that
the workflow calls. Local Git commits are fine and do not create a PR. This
proves the container execution path before GitHub needs the new workflow.

Qualify the immutable candidate on an approved GPU machine. The host needs a
compatible NVIDIA driver, Docker, NVIDIA Container Toolkit and Python 3.10+;
it does not need your Python environment or host CUDA toolkit. Reserve GPUs
with the team **before** adding them to `COLOSSAL_GPU_POOL`. An idle GPU is not
automatically available for CI use. All executors must use the same agreed lock.

```bash
# Required pre-agreed settings. Missing values stop the example rather than
# using every idle GPU or inventing a second, ineffective lock.
: "${COLOSSAL_CI_IMAGE:?Set the candidate's full local sha256:image-ID}"
: "${COLOSSAL_GPU_POOL:?Set authorized physical GPU indices, e.g. 4 for one allocated GPU}"
: "${COLOSSAL_GPU_LOCK:?Set the existing shared CI GPU lock file}"
: "${CI_QUALIFICATION_ROOT:?Set an approved personal directory for source and reports}"
export COLOSSAL_GPU_POOL
export CI_SOURCE_SHA="$(git rev-parse HEAD)"
export FAST_MODE=0

# Test committed source only. No local .so, editable install or .git credentials
# are carried into the container. Commit the proposed changes before this step.
mkdir -p "$CI_QUALIFICATION_ROOT"
CI_SOURCE_DIR="$(mktemp -d "$CI_QUALIFICATION_ROOT/source.XXXXXX")"
git archive "$CI_SOURCE_SHA" | tar -x -C "$CI_SOURCE_DIR"
docker image inspect --format '{{.Id}}' "$COLOSSAL_CI_IMAGE"

# First run FP8 batch 4a through the same allocation, container and cleanup code
# used by Actions. Logs, JUnit XML and runtime/result JSON go into this directory.
python3 -B "$CI_SOURCE_DIR/.github/scripts/ci/run_gpu_batch.py" \
  --image "$COLOSSAL_CI_IMAGE" --batch 4a \
  --lock "$COLOSSAL_GPU_LOCK" \
  --output "$CI_QUALIFICATION_ROOT/results-$CI_SOURCE_SHA/batch-4a"
```

Repeat the last command with batch `3` (AOT build + Adam), `4b` (inference CUDA),
then `2` and `5` (single-/two-GPU core); batch 5 requires two authorized GPUs.
For cancellation qualification, interrupt **only this test command**, then
check its reported container name is absent from `docker ps -a` and that no
test-owned GPU processes remain. Do not kill or prune unrelated jobs.

Batch 4a is the FP8 cast/linear/hook regression and is unaffected by `FAST_TEST`.
Batch 3 (Adam kernels) and 4b (inference CUDA kernels) are also unaffected. Layer 2
selects batches `2 4a 5` with `FAST_MODE=1`, which reduces model matrices in 2/5.
Layer 3 defaults to `FAST_MODE=0`; the batch script **unsets** `FAST_TEST` instead
of setting it to zero. Testmon selection is not enabled.

The executor holds the host lock through named-container cleanup, filters
devices by UUID and supplies logical `0,1,...` inside the container. Source is
mounted read-only and copied into the container; reports are the only writable
bind, owned by the runner UID/GID. A lightweight private venv references image
dependencies without copying Torch or modifying the base venv. All compiler
caches are initially disposable. Batch 3 compiles AOT with `BUILD_EXT=1`; other
batches exercise JIT as needed. No old ColossalAI binary is reused.

Before installation, the container compares its embedded Dockerfile, context
ignore file, locks and verifier with the selected source. A mismatch fails with
an explicit rebuild message rather than testing a dependency-upgrade PR against
the old environment. Documentation-only and normal source changes reuse the
image. When changing the recipe, qualify the candidate before pointing CI at it.

After candidate qualification, set repository variables documented in the
[workflow README](../../.github/workflows/README.md#container-execution-and-image-rollout),
then open the **one combined PR** and exercise its workflow before merging the
switch. Start with a Draft PR: its separate container-smoke check runs only 4a
using `COLOSSAL_CI_SMOKE_GPU_POOL`, not the main `COLOSSAL_GPU_POOL`. Missing
smoke authorization fails closed. Mark the same PR Ready only after arranging
the resources for the full Layer 2 batches `2 4a 5`; smoke success does not
replace those checks. No dependency-image rebuild is needed for routing-only
workflow/script changes because none of the embedded image inputs changes.
The runner checks the preloaded local image before acquiring GPUs; it
does not pull, push or repeatedly reload archives. A local image on another
daemon is insufficient, even when a tar is present on a shared filesystem.
Preload every machine eligible for the runner label or restrict that label to
the provisioned host. Source download and artifact upload still use GitHub.

Pre-PR script runs do not prove GitHub event dispatch, permissions, Checks or
artifact upload. Verify those on the single PR, and make any further fixes on
that same branch. Do not merge first just to discover whether containers work.

This recipe does not automatically publish images, configure Docker on the
shared server, or change GitHub required checks. Build/load once per dependency
profile change; code-only PRs reuse the image. Keep the local image ID, optional
archive checksum, base digest, source SHA, build log and qualification reports together. A recipe
or successful CPU check alone is not proof that the GPU CI profile passes.

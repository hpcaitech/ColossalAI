# syntax=docker/dockerfile:1
# Build from the repository root. No GPU is needed to build this dependency image.
ARG CUDA_IMAGE=nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv \
    build-essential cmake ninja-build git curl ca-certificates \
    libaio-dev libibverbs1 libibverbs-dev ibverbs-providers librdmacm1 rdma-core \
    && rm -rf /var/lib/apt/lists/*
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    MAX_JOBS=4

FROM base AS dependencies
ARG PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
COPY docker/torch213/requirements*.lock /opt/ci/
# The optional wheelhouse is a read-only build input, never a final-image layer.
# Core wheels are either supplied by the builder or fetched from the Torch index.
RUN --mount=type=bind,source=docker/torch213/wheelhouse,target=/wheelhouse,readonly \
    --mount=type=cache,target=/root/.cache/pip \
    if find /wheelhouse -maxdepth 1 -name '*.whl' -print -quit | grep -q .; then \
      python -m pip install --no-deps --no-index --find-links=/wheelhouse \
        --require-hashes -r /opt/ci/requirements-torch.lock; \
    else \
      python -m pip install --no-deps --index-url "${TORCH_INDEX_URL}" \
        --require-hashes -r /opt/ci/requirements-torch.lock; \
    fi && \
    python -m pip install --no-deps --index-url "${PIP_INDEX_URL}" \
      -r /opt/ci/requirements.lock && \
    python -m pip check

FROM base AS ci
ARG SOURCE_COMMIT=unknown
LABEL org.opencontainers.image.title="ColossalAI Torch 2.13 CI dependencies" \
    org.opencontainers.image.source="https://github.com/hpcaitech/ColossalAI" \
    org.opencontainers.image.revision="${SOURCE_COMMIT}"
COPY --from=dependencies /opt/venv /opt/venv
COPY docker/torch213/requirements*.lock docker/torch213/verify_image.py /opt/ci/
WORKDIR /workspace
# This checks metadata, imports and CPU operations; CUDA remains hidden.
RUN NVIDIA_VISIBLE_DEVICES=void CUDA_VISIBLE_DEVICES="" python /opt/ci/verify_image.py
CMD ["/bin/bash"]

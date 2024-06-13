FROM hpcaitech/cuda-conda:12.1

# metainformation
LABEL org.opencontainers.image.source = "https://github.com/hpcaitech/ColossalAI"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name = "docker.io/library/hpcaitech/cuda-conda:12.1"

# enable passwordless ssh
RUN mkdir ~/.ssh && \
    printf "Host * \n    ForwardAgent yes\nHost *\n    StrictHostKeyChecking no" > ~/.ssh/config && \
    ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

# enable RDMA support
RUN apt-get update && \
    apt-get install -y infiniband-diags perftest ibverbs-providers libibumad3 libibverbs1 libnl-3-200 libnl-route-3-200 librdmacm1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install torch
RUN conda install -y python==3.10 && conda install -y pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# install ninja
RUN apt-get update && \
    apt-get install -y --no-install-recommends ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install apex
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    git checkout a7de60 && \
    pip install packaging && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# install colossalai
ARG VERSION=main
RUN git clone -b ${VERSION} https://github.com/hpcaitech/ColossalAI.git \
    && cd ./ColossalAI \
    && BUILD_EXT=1 pip install -v . \
    && rm -rf colossalai

# install tensornvme
RUN conda install -y cmake && \
    apt update -y && apt install -y libaio-dev && \
    pip install -v git+https://github.com/hpcaitech/TensorNVMe.git

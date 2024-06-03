FROM hpcaitech/cuda-conda:11.3

# metainformation
LABEL org.opencontainers.image.source = "https://github.com/hpcaitech/ColossalAI"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name = "docker.io/library/hpcaitech/cuda-conda:11.3"

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
RUN conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# install ninja
RUN apt-get update && \
    apt-get install -y --no-install-recommends ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install apex
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    git checkout 91fcaa && \
    pip install packaging && \
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" ./

# install colossalai
ARG VERSION=main
RUN git clone -b ${VERSION} https://github.com/hpcaitech/ColossalAI.git \
    && cd ./ColossalAI \
    && BUILD_EXT=1 pip install -v --no-cache-dir .

# install titans
RUN pip install --no-cache-dir titans

# install tensornvme
RUN conda install -y cmake && \
    git clone https://github.com/hpcaitech/TensorNVMe.git && \
    cd TensorNVMe && \
    apt update -y && apt install -y libaio-dev && \
    pip install -r requirements.txt && \
    pip install -v --no-cache-dir .

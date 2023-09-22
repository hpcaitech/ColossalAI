# FROM hpcaitech/cuda-conda:11.6
FROM hpcaitech/pytorch-cuda:1.13.0-11.6.0

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

# install ninja
RUN apt-get update && \
    apt-get install -y --no-install-recommends ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install colossalai
ARG VERSION=main
RUN git clone -b ${VERSION} https://github.com/hpcaitech/ColossalAI.git \
    && cd ./ColossalAI \
    && git checkout 3e05c07bb8921f2a8f9736b6f6673d4e9f1697d0 \
    && CUDA_EXT=1 pip install -v --no-cache-dir .

# install titans
RUN pip install --no-cache-dir titans

# install triton
RUN pip install --no-cache-dir triton==2.0.0.dev20221202

# install torchserve
ARG VERSION=master
RUN git clone -b ${VERSION} https://github.com/pytorch/serve.git \
    && cd ./serve \
    && python ./ts_scripts/install_dependencies.py --cuda=cu116 \
    && pip install torchserve torch-model-archiver torch-workflow-archiver

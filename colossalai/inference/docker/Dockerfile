FROM hpcaitech/cuda-conda:11.8

# metainformation
LABEL org.opencontainers.image.source = "https://github.com/hpcaitech/ColossalAI"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name = "docker.io/library/hpcaitech/cuda-conda:11.3"

# enable RDMA support
RUN apt-get update && \
    apt-get install -y infiniband-diags perftest ibverbs-providers libibumad3 libibverbs1 libnl-3-200 libnl-route-3-200 librdmacm1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN git clone --recursive https://github.com/hpcaitech/ColossalAI
WORKDIR ColossalAI && pip install -e .

# install packages from requirements-infer.txt
WORKDIR requirements
RUN pip install triton
RUN pip install --upgrade pip
COPY requirements-infer.txt requirements-infer.txt
RUN pip install -r requirements-infer.txt


# install flash-attention
WORKDIR /workspace
RUN git clone --recursive https://github.com/Dao-AILab/flash-attention
RUN cd flash-attention && pip install -e .
RUN cd ../




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

RUN pip install triton 
RUN pip install git+https://github.com/ModelTC/lightllm.git@ece7b43f8a6dfa74027adc77c2c176cff28c76c8

cd /workspace
RUN git clone --recursive https://github.com/hpcaitech/ColossalAI
RUN cd ColossalAI && pip install -e .



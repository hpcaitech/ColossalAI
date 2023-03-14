FROM hpcaitech/cuda-conda:11.3

# metainformation
LABEL org.opencontainers.image.source = "https://github.com/hpcaitech/ColossalAI"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name = "docker.io/library/hpcaitech/cuda-conda:11.3"

# install torch
RUN conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# install apex
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install packaging && \
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" ./

# install colossalai
RUN git clone https://github.com/hpcaitech/ColossalAI.git \
    && cd ./ColossalAI \
    && CUDA_EXT=1 pip install -v --no-cache-dir .

# install titans
RUN pip install --no-cache-dir titans

# install tensornvme
RUN conda install cmake && \
    git clone https://github.com/hpcaitech/TensorNVMe.git && \
    cd TensorNVMe && \
    pip install -r requirements.txt && \
    pip install -v --no-cache-dir .

FROM hpcaitech/cuda-conda:11.3

# install torch
RUN conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# install apex
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# install colossalai
RUN git clone https://github.com/hpcaitech/ColossalAI.git \
 && cd ./ColossalAI \
 && pip install -v --no-cache-dir .

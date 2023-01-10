FROM hpcaitech/pytorch-cuda:1.12.0-11.3.0

# install torch
# RUN conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# install apex
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" ./

# install colossalai
# RUN git clone https://github.com/hpcaitech/ColossalAI.git \
#     && cd ./ColossalAI \
#     && pip install -v --no-cache-dir .

RUN pip install colossalai==0.1.12+torch1.12cu11.3 -f https://release.colossalai.org


# install our lightning, it will be merged to Lightning official repo.
RUN git clone https://github.com/1SAA/lightning.git && \
    cd lightning && \
    git checkout strategy/colossalai && \
    export PACKAGE_NAME=pytorch && \
    pip install --no-cache-dir .

# install titans
RUN pip install --no-cache-dir titans

RUN git clone https://github.com/hpcaitech/ColossalAI.git  && \
    cd ./ColossalAI/examples/images/diffusion && \
    pip install -r requirements.txt && \
    pip install --no-cache-dir transformers==4.19.2 diffusers invisible-watermark

# install tensornvme
# RUN conda install cmake && \
#     git clone https://github.com/hpcaitech/TensorNVMe.git && \
#     cd TensorNVMe && \
#     pip install -r requirements.txt && \
#     pip install -v --no-cache-dir .

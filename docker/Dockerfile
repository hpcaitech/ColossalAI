FROM nvcr.io/nvidia/pytorch:21.07-py3

# install dependencies
RUN pip install -U pip setuptools wheel \
 && pip install pytest tensorboard deepspeed apex

# install colossalai
RUN git clone https://github.com/hpcaitech/ColossalAI.git \
 && cd ./ColossalAI \
 && pip install -v --no-cache-dir --global-option="--cuda_ext" .

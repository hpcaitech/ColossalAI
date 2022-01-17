FROM nvcr.io/nvidia/pytorch:21.07-py3

# install dependencies
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
 && pip install -U pip setuptools wheel \
 && pip install pytest tensorboard deepspeed apex

# install colossalai
RUN git clone https://github.com/hpcaitech/ColossalAI.git \
 && cd ./ColossalAI \
 && pip install -v --no-cache-dir --global-option="--cuda_ext" .

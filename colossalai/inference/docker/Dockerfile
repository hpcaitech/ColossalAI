FROM tensorrt_llm/release:latest

# metainformation
LABEL org.opencontainers.image.source = "https://github.com/hpcaitech/ColossalAI"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name = "tensorrt_llm/release:latest"

# install flash-attention
WORKDIR /workspace
RUN git clone --recursive https://github.com/Dao-AILab/flash-attention
RUN cd flash-attention && pip install -e .
RUN cd ../

# install colossalai 
WORKDIR /workspace
RUN git clone --recursive https://github.com/hpcaitech/ColossalAI
RUN cd ColossalAI && pip install -e .
RUN pip install triton

WORKDIR /workspace/ColossalAI/requirements
RUN pip install -r requirements-infer.txt




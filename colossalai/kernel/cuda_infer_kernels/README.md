All of Ops are created for colossal-ai, and it will be used for inference platform of colossal-ai.

The flash attention v2 kernel has been extracted from [the original repo](https://github.com/Dao-AILab/flash-attention) into this repo to make it easier to integrate into a third-party project. In particular, the dependency on libtorch was removed. Also we modified some of ops from it to make it be suitable for 3rd party libs.

As a consquence, dropout is not supported (since the original code uses randomness provided by libtorch). Also, only forward is supported for now.

How to Install:

```
# please install cmake, ninja and torch >= 1.12.0 before install ColossalTorch 
git clone --recursive https://github.com/tiandiao123/ColossalTorch
cd ColossalTorch
pip install -e .
```
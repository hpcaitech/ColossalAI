# Setup

Requirements:
- PyTorch >= 1.11 and PyTorch <= 2.1
- Python >= 3.7
- CUDA >= 11.0
- [NVIDIA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) >= 7.0 (V100/RTX20 and higher)
- Linux OS

If you encounter any problem about installation, you may want to raise an [issue](https://github.com/hpcaitech/ColossalAI/issues/new/choose) in this repository.


## Download From PyPI

You can install Colossal-AI with

```shell
pip install colossalai
```

**Note: only Linux is supported for now**

If you want to build PyTorch extensions during installation, you can use the command below. Otherwise, the PyTorch extensions will be built during runtime.

```shell
CUDA_EXT=1 pip install colossalai
```


## Download From Source

> The version of Colossal-AI will be in line with the main branch of the repository. Feel free to raise an issue if you encounter any problem.

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# install dependency
pip install -r requirements/requirements.txt

# install colossalai
CUDA_EXT=1 pip install .
```

If you don't want to install and enable CUDA kernel fusion (compulsory installation when using fused optimizer), just don't specify the `CUDA_EXT`:

```shell
pip install .
```

For Users with CUDA 10.2, you can still build ColossalAI from source. However, you need to manually download the cub library and copy it to the corresponding directory.

```bash
# clone the repository
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# download the cub library
wget https://github.com/NVIDIA/cub/archive/refs/tags/1.8.0.zip
unzip 1.8.0.zip
cp -r cub-1.8.0/cub/ colossalai/kernel/cuda_native/csrc/kernels/include/

# install
CUDA_EXT=1 pip install .
```

<!-- doc-test-command: echo "installation.md does not need test" -->

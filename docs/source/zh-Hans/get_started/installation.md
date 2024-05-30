# 安装

环境要求:

- PyTorch >= 2.1
- Python >= 3.7
- CUDA >= 11.0
- [NVIDIA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) >= 7.0 (V100/RTX20 and higher)
- Linux OS

如果你遇到安装问题，可以向本项目 [反馈](https://github.com/hpcaitech/ColossalAI/issues/new/choose)。

## 从PyPI上安装

你可以PyPI上使用以下命令直接安装Colossal-AI。

```shell
pip install colossalai
```

**注：现在只支持Linux。**

如果你想同时安装PyTorch扩展的话，可以添加`BUILD_EXT=1`。如果不添加的话，PyTorch扩展会在运行时自动安装。

```shell
BUILD_EXT=1 pip install colossalai
```

## 从源安装

> 此文档将与版本库的主分支保持一致。如果您遇到任何问题，欢迎给我们提 issue。

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# install dependency
pip install -r requirements/requirements.txt

# install colossalai
BUILD_EXT=1 pip install .
```

如果您不想安装和启用 CUDA 内核融合（使用融合优化器时强制安装），您可以不添加`BUILD_EXT=1`：

```shell
pip install .
```

如果您在使用CUDA 10.2，您仍然可以从源码安装ColossalAI。但是您需要手动下载cub库并将其复制到相应的目录。

```bash
# clone the repository
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# download the cub library
wget https://github.com/NVIDIA/cub/archive/refs/tags/1.8.0.zip
unzip 1.8.0.zip
cp -r cub-1.8.0/cub/ colossalai/kernel/cuda_native/csrc/kernels/include/

# install
BUILD_EXT=1 pip install .
```

<!-- doc-test-command: echo "installation.md does not need test" -->

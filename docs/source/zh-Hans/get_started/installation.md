# 安装

环境要求:

- PyTorch >= 1.11 (PyTorch 2.x 正在适配中)
- Python >= 3.7
- CUDA >= 11.0

如果你遇到安装问题，可以向本项目 [反馈](https://github.com/hpcaitech/ColossalAI/issues/new/choose)。

## 从PyPI上安装

你可以PyPI上使用以下命令直接安装Colossal-AI。

```shell
pip install colossalai
```

**注：现在只支持Linux。**

如果你想同时安装PyTorch扩展的话，可以添加`CUDA_EXT=1`。如果不添加的话，PyTorch扩展会在运行时自动安装。

```shell
CUDA_EXT=1 pip install colossalai
```

## 从源安装

> 此文档将与版本库的主分支保持一致。如果您遇到任何问题，欢迎给我们提 issue :)

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# install dependency
pip install -r requirements/requirements.txt

# install colossalai
pip install .
```

如果您不想安装和启用 CUDA 内核融合（使用融合优化器时强制安装）：

```shell
NO_CUDA_EXT=1 pip install .
```

<!-- doc-test-command: echo "installation.md does not need test" -->

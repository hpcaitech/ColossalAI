# Colossal-AI
<div id="top" align="center">

   [![logo](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/Colossal-AI_logo.png)](https://www.colossalai.org/)

   一个整合高效并行技术的AI大模型训练系统。

   <h3> <a href="https://arxiv.org/abs/2110.14883"> 论文 </a> | 
   <a href="https://www.colossalai.org/"> 文档 </a> | 
   <a href="https://github.com/hpcaitech/ColossalAI-Examples"> 例程 </a> |   
   <a href="https://github.com/hpcaitech/ColossalAI/discussions"> 论坛 </a> | 
   <a href="https://medium.com/@hpcaitech"> 博客 </a></h3>

   [![Build](https://github.com/hpcaitech/ColossalAI/actions/workflows/build.yml/badge.svg)](https://github.com/hpcaitech/ColossalAI/actions/workflows/build.yml)
   [![Documentation](https://readthedocs.org/projects/colossalai/badge/?version=latest)](https://colossalai.readthedocs.io/en/latest/?badge=latest)
   [![CodeFactor](https://www.codefactor.io/repository/github/hpcaitech/colossalai/badge)](https://www.codefactor.io/repository/github/hpcaitech/colossalai)
   [![HuggingFace badge](https://img.shields.io/badge/%F0%9F%A4%97HuggingFace-Join-yellow)](https://huggingface.co/hpcai-tech)
   [![slack badge](https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp)](https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-z7b26eeb-CBp7jouvu~r0~lcFzX832w)
   [![WeChat badge](https://img.shields.io/badge/微信-加入-green?logo=wechat&amp)](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png)

   | [English](README.md) | [中文](README-zh-Hans.md) |

</div>


## 目录
<ul>
 <li><a href="#特点">特点</a> </li>
 <li>
   <a href="#展示样例">展示样例</a> 
   <ul>
     <li><a href="#ViT">ViT</a></li>
     <li><a href="#GPT-3">GPT-3</a></li>
     <li><a href="#GPT-2">GPT-2</a></li>
     <li><a href="#BERT">BERT</a></li>
   </ul>
 </li>

 <li>
   <a href="#安装">安装</a>
   <ul>
     <li><a href="#PyPI">PyPI</a></li>
     <li><a href="#从源代码安装">从源代码安装</a></li>
   </ul>
 </li>
 <li><a href="#使用-Docker">使用 Docker</a></li>
 <li><a href="#社区">社区</a></li>
 <li><a href="#做出贡献">做出贡献</a></li>
 <li><a href="#快速预览">快速预览</a></li>
   <ul>
     <li><a href="#几行代码开启分布式训练">几行代码开启分布式训练</a></li>
     <li><a href="#构建一个简单的2维并行模型">构建一个简单的2维并行模型</a></li>
   </ul>
 <li><a href="#引用我们">引用我们</a></li>
</ul>

## 特点

Colossal-AI为您提供了一系列并行训练组件。我们的目标是让您的分布式AI模型训练像普通的单GPU模型一样简单。我们提供的友好工具可以让您在几行代码内快速开始分布式训练。

- 数据并行
- 流水线并行
- 1维, 2维, 2.5维, 3维张量并行 
- 序列并行
- 友好的trainer和engine
- 可扩展新的并行方式
- 混合精度
- 零冗余优化器 (ZeRO)

<p align="right">(<a href="#top">返回顶端</a>)</p>

## 展示样例
### ViT

<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/ViT.png" width="450" />

- 14倍批大小和5倍训练速度（张量并行=64）

### GPT-3
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT3.png" width=700/>

- 释放 50% GPU 资源占用, 或 10.7% 加速

### GPT-2
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT2.png" width=800/>

- 降低11倍GPU显存占用，或超线性扩展（张量并行）

<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/Colossal-AI%20with%20ZeRO.jpg" width=393>

- 能训练接近11倍大小的模型（ZeRO） 

### BERT
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/BERT.png" width=800/>

- 2倍训练速度，或1.5倍序列长度

请访问我们的[文档和教程](https://www.colossalai.org/)以了解详情。

<p align="right">(<a href="#top">返回顶端</a>)</p>

## 安装

### PyPI

```bash
pip install colossalai
```
该命令将会安装CUDA extension，如果你已安装CUDA, NVCC和torch。 

如果你不想安装CUDA extension, 可在命令中添加`--global-option="--no_cuda_ext"`, 例如:
```bash
pip install colossalai --global-option="--no_cuda_ext"
```

如果你想使用`ZeRO`, 你可以使用:
```bash
pip install colossalai[zero]
```

### 从源代码安装

> Colossal-AI的版本将与该项目的主分支保持一致。欢迎通过issue反馈你遇到的任何问题 :)

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI
# 安装依赖
pip install -r requirements/requirements.txt

# 安装 colossalai
pip install .
```

如果你不想安装和使用CUDA kernel fusion (使用fused优化器需安装):

```shell
pip install --global-option="--no_cuda_ext" .
```

<p align="right">(<a href="#top">返回顶端</a>)</p>

## 使用 Docker

运行以下命令从我们提供的docker文件中建立docker镜像。

```bash
cd ColossalAI
docker build -t colossalai ./docker
```

运行以下命令从以交互式启动docker镜像.

```bash
docker run -ti --gpus all --rm --ipc=host colossalai bash
```

<p align="right">(<a href="#top">返回顶端</a>)</p>

## 社区
欢迎通过[论坛](https://github.com/hpcaitech/ColossalAI/discussions),
[Slack](https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-z7b26eeb-CBp7jouvu~r0~lcFzX832w),
或[微信](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png "qrcode")加入Colossal-AI社区，与我们分享你的建议和问题。


## 做出贡献

欢迎为该项目做出贡献，请参阅[贡献指南](./CONTRIBUTING.md)。

真诚感谢所有贡献者！

<a href="https://github.com/hpcaitech/ColossalAI/graphs/contributors"><img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/contributor_avatar.png" width="800px"></a>

*贡献者头像的展示顺序是随机的。*

<p align="right">(<a href="#top">返回顶端</a>)</p>

## 快速预览

### 几行代码开启分布式训练

```python
import colossalai
from colossalai.utils import get_dataloader


# my_config可以是config文件的路径或字典对象
# 'localhost' 仅适用于单节点，在多节点时需指明节点名
colossalai.launch(
    config=my_config,
    rank=rank,
    world_size=world_size,
    backend='nccl',
    port=29500,
    host='localhost'
)

# 构建模型
model = ...

# 构建数据集, dataloader会默认处理分布式数据sampler
train_dataset = ...
train_dataloader = get_dataloader(dataset=dataset,
                                shuffle=True
                                )


# 构建优化器
optimizer = ...

# 构建损失函数
criterion = ...

# 初始化colossalai
engine, train_dataloader, _, _ = colossalai.initialize(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_dataloader=train_dataloader
)

# 开始训练
engine.train()
for epoch in range(NUM_EPOCHS):
    for data, label in train_dataloader:
        engine.zero_grad()
        output = engine(data)
        loss = engine.criterion(output, label)
        engine.backward(loss)
        engine.step()

```

### 构建一个简单的2维并行模型

假设我们有一个非常巨大的MLP模型，它巨大的hidden size使得它难以被单个GPU容纳。我们可以将该模型的权重以二维网格的形式分配到多个GPU上，且保持你熟悉的模型构建方式。

```python
from colossalai.nn import Linear2D
import torch.nn as nn


class MLP_2D(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_1 = Linear2D(in_features=1024, out_features=16384)
        self.linear_2 = Linear2D(in_features=16384, out_features=1024)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x

```

<p align="right">(<a href="#top">返回顶端</a>)</p>

## 引用我们

```
@article{bian2021colossal,
  title={Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training},
  author={Bian, Zhengda and Liu, Hongxin and Wang, Boxiang and Huang, Haichen and Li, Yongbin and Wang, Chuanrui and Cui, Fan and You, Yang},
  journal={arXiv preprint arXiv:2110.14883},
  year={2021}
}
```

<p align="right">(<a href="#top">返回顶端</a>)</p>

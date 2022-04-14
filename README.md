# Colossal-AI
<div id="top" align="center">

   [![logo](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/Colossal-AI_logo.png)](https://www.colossalai.org/)

   An integrated large-scale model training system with efficient parallelization techniques.

   <h3> <a href="https://arxiv.org/abs/2110.14883"> Paper </a> | 
   <a href="https://www.colossalai.org/"> Documentation </a> | 
   <a href="https://github.com/hpcaitech/ColossalAI-Examples"> Examples </a> |   
   <a href="https://github.com/hpcaitech/ColossalAI/discussions"> Forum </a> | 
   <a href="https://medium.com/@hpcaitech"> Blog </a></h3>

   [![Build](https://github.com/hpcaitech/ColossalAI/actions/workflows/build.yml/badge.svg)](https://github.com/hpcaitech/ColossalAI/actions/workflows/build.yml)
   [![Documentation](https://readthedocs.org/projects/colossalai/badge/?version=latest)](https://colossalai.readthedocs.io/en/latest/?badge=latest)
   [![CodeFactor](https://www.codefactor.io/repository/github/hpcaitech/colossalai/badge)](https://www.codefactor.io/repository/github/hpcaitech/colossalai)
   [![HuggingFace badge](https://img.shields.io/badge/%F0%9F%A4%97HuggingFace-Join-yellow)](https://huggingface.co/hpcai-tech)
   [![slack badge](https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp)](https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-z7b26eeb-CBp7jouvu~r0~lcFzX832w)
   [![WeChat badge](https://img.shields.io/badge/微信-加入-green?logo=wechat&amp)](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png)
   

   | [English](README.md) | [中文](README-zh-Hans.md) |

</div>

## Table of Contents
<ul>
 <li><a href="#Why-Colossal-AI">Why Colossal-AI</a> </li>
 <li><a href="#Features">Features</a> </li>
 <li>
   <a href="#Demo">Demo</a> 
   <ul>
     <li><a href="#ViT">ViT</a></li>
     <li><a href="#GPT-3">GPT-3</a></li>
     <li><a href="#GPT-2">GPT-2</a></li>
     <li><a href="#BERT">BERT</a></li>
     <li><a href="#PaLM">PaLM</a></li>
   </ul>
 </li>

 <li>
   <a href="#Installation">Installation</a>
   <ul>
     <li><a href="#PyPI">PyPI</a></li>
     <li><a href="#Install-From-Source">Install From Source</a></li>
   </ul>
 </li>
 <li><a href="#Use-Docker">Use Docker</a></li>
 <li><a href="#Community">Community</a></li>
 <li><a href="#contributing">Contributing</a></li>
 <li><a href="#Quick-View">Quick View</a></li>
   <ul>
     <li><a href="#Start-Distributed-Training-in-Lines">Start Distributed Training in Lines</a></li>
     <li><a href="#Write-a-Simple-2D-Parallel-Model">Write a Simple 2D Parallel Model</a></li>
   </ul>
 <li><a href="#Cite-Us">Cite Us</a></li>
</ul>

## Why Colossal-AI
<div align="center">
   <a href="https://youtu.be/KnXSfjqkKN0">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/JamesDemmel_Colossal-AI.png" width="600" />
   </a>

   Prof. James Demmel (UC Berkeley): Colossal-AI makes distributed training efficient, easy and scalable.
</div>

<p align="right">(<a href="#top">back to top</a>)</p>

## Features

Colossal-AI provides a collection of parallel training components for you. We aim to support you to write your
distributed deep learning models just like how you write your model on your laptop. We provide user-friendly tools to kickstart
distributed training in a few lines.

- Parallelism strategies
  - Data Parallelism
  - Pipeline Parallelism
  - 1D, [2D](https://arxiv.org/abs/2104.05343), [2.5D](https://arxiv.org/abs/2105.14500), 3D Tensor parallelism
  - [Sequence parallelism](https://arxiv.org/abs/2105.13120)
  - [Zero Redundancy Optimizer (ZeRO)](https://arxiv.org/abs/2108.05818)

- Heterogeneous Memory Menagement 
  - [PatrickStar](https://arxiv.org/abs/2108.05818)

- Friendly Usage
  - Configuration file based

<p align="right">(<a href="#top">back to top</a>)</p>

## Demo
### ViT
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/ViT.png" width="450" />
</p>

- 14x larger batch size, and 5x faster training for Tensor Parallelism = 64

### GPT-3
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT3.png" width=700/>
</p>

- Save 50% GPU resources, and 10.7% acceleration

### GPT-2
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT2.png" width=800/>

- 11x lower GPU memory consumption, and superlinear scaling efficiency with Tensor Parallelism

<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/(updated)GPT-2.png" width=800>

- 24x larger model size on the same hardware
- over 3x acceleration
### BERT
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/BERT.png" width=800/>

- 2x faster training, or 50% longer sequence length

### PaLM
- [PaLM-colossalai](https://github.com/hpcaitech/PaLM-colossalai): Scalable implementation of Google's Pathways Language Model ([PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html)).

Please visit our [documentation and tutorials](https://www.colossalai.org/) for more details.

<p align="right">(<a href="#top">back to top</a>)</p>

## Installation

### PyPI

```bash
pip install colossalai
```
This command will install CUDA extension if your have installed CUDA, NVCC and torch. 

If you don't want to install CUDA extension, you should add `--global-option="--no_cuda_ext"`, like:
```bash
pip install colossalai --global-option="--no_cuda_ext"
```

### Install From Source

> The version of Colossal-AI will be in line with the main branch of the repository. Feel free to create an issue if you encounter any problems. :-)

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI
# install dependency
pip install -r requirements/requirements.txt

# install colossalai
pip install .
```

If you don't want to install and enable CUDA kernel fusion (compulsory installation when using fused optimizer):

```shell
pip install --global-option="--no_cuda_ext" .
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Use Docker

Run the following command to build a docker image from Dockerfile provided.

```bash
cd ColossalAI
docker build -t colossalai ./docker
```

Run the following command to start the docker container in interactive mode.

```bash
docker run -ti --gpus all --rm --ipc=host colossalai bash
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Community

Join the Colossal-AI community on [Forum](https://github.com/hpcaitech/ColossalAI/discussions),
[Slack](https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-z7b26eeb-CBp7jouvu~r0~lcFzX832w),
and [WeChat](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png "qrcode") to share your suggestions, feedback, and questions with our engineering team.

## Contributing

If you wish to contribute to this project, please follow the guideline in [Contributing](./CONTRIBUTING.md).

Thanks so much to all of our amazing contributors!

<a href="https://github.com/hpcaitech/ColossalAI/graphs/contributors"><img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/contributor_avatar.png" width="800px"></a>

*The order of contributor avatars is randomly shuffled.*

<p align="right">(<a href="#top">back to top</a>)</p>

## Quick View

### Start Distributed Training in Lines

```python
import colossalai
from colossalai.utils import get_dataloader


# my_config can be path to config file or a dictionary obj
# 'localhost' is only for single node, you need to specify
# the node name if using multiple nodes
colossalai.launch(
    config=my_config,
    rank=rank,
    world_size=world_size,
    backend='nccl',
    port=29500,
    host='localhost'
)

# build your model
model = ...

# build you dataset, the dataloader will have distributed data
# sampler by default
train_dataset = ...
train_dataloader = get_dataloader(dataset=dataset,
                                shuffle=True
                                )


# build your optimizer
optimizer = ...

# build your loss function
criterion = ...

# initialize colossalai
engine, train_dataloader, _, _ = colossalai.initialize(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_dataloader=train_dataloader
)

# start training
engine.train()
for epoch in range(NUM_EPOCHS):
    for data, label in train_dataloader:
        engine.zero_grad()
        output = engine(data)
        loss = engine.criterion(output, label)
        engine.backward(loss)
        engine.step()

```

### Write a Simple 2D Parallel Model

Let's say we have a huge MLP model and its very large hidden size makes it difficult to fit into a single GPU. We can
then distribute the model weights across GPUs in a 2D mesh while you still write your model in a familiar way.

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

<p align="right">(<a href="#top">back to top</a>)</p>

## Cite Us

```
@article{bian2021colossal,
  title={Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training},
  author={Bian, Zhengda and Liu, Hongxin and Wang, Boxiang and Huang, Haichen and Li, Yongbin and Wang, Chuanrui and Cui, Fan and You, Yang},
  journal={arXiv preprint arXiv:2110.14883},
  year={2021}
}
```

<p align="right">(<a href="#top">back to top</a>)</p>
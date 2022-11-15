# Colossal-AI
<div id="top" align="center">

   [![logo](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/Colossal-AI_logo.png)](https://www.colossalai.org/)

   Colossal-AI: 一个面向大模型时代的通用深度学习系统

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

## 新闻

* [2022/11] [Diffusion Pretraining and Hardware Fine-Tuning Can Be Almost 7X Cheaper](https://medium.com/@yangyou_berkeley/diffusion-pretraining-and-hardware-fine-tuning-can-be-almost-7x-cheaper-85e970fe207b)
* [2022/10] [Use a Laptop to Analyze 90% of Proteins, With a Single-GPU Inference Sequence Exceeding 10,000](https://medium.com/@yangyou_berkeley/use-a-laptop-to-analyze-90-of-proteins-with-a-single-gpu-inference-sequence-exceeding-10-000-4c8f0a389cd)
* [2022/10] [Embedding Training With 1% GPU Memory and 100 Times Less Budget for Super-Large Recommendation Model](https://medium.com/@yangyou_berkeley/embedding-training-with-1-gpu-memory-and-10-times-less-budget-an-open-source-solution-for-6b4c3aba07a8)
* [2022/09] [HPC-AI Tech Completes $6 Million Seed and Angel Round Fundraising](https://medium.com/@hpcaitech/hpc-ai-tech-completes-6-million-seed-and-angel-round-fundraising-led-by-bluerun-ventures-in-the-892468cc2b02)
* [2022/07] [Colossal-AI Seamlessly Accelerates Large Models at Low Costs with Hugging Face](https://medium.com/@yangyou_berkeley/colossal-ai-seamlessly-accelerates-large-models-at-low-costs-with-hugging-face-4d1a887e500d)


## 目录
<ul>
 <li><a href="#为何选择-Colossal-AI">为何选择 Colossal-AI</a> </li>
 <li><a href="#特点">特点</a> </li>
 <li>
   <a href="#并行训练样例展示">并行训练样例展示</a> 
   <ul>
     <li><a href="#ViT">ViT</a></li>
     <li><a href="#GPT-3">GPT-3</a></li>
     <li><a href="#GPT-2">GPT-2</a></li>
     <li><a href="#BERT">BERT</a></li>
     <li><a href="#PaLM">PaLM</a></li>
     <li><a href="#OPT">OPT</a></li>
     <li><a href="#推荐系统模型">推荐系统模型</a></li>
   </ul>
 </li>
<li>
   <a href="#单GPU训练样例展示">单GPU训练样例展示</a> 
   <ul>
     <li><a href="#GPT-2-Single">GPT-2</a></li>
     <li><a href="#PaLM-Single">PaLM</a></li>
   </ul>
 </li>
<li>
   <a href="#推理-Energon-AI-样例展示">推理 (Energon-AI) 样例展示</a> 
   <ul>
     <li><a href="#GPT-3-Inference">GPT-3</a></li>
     <li><a href="#OPT-Serving">1750亿参数OPT在线推理服务</a></li>
   </ul>
 </li>
<li>
   <a href="#Colossal-AI-in-the-Real-World">Colossal-AI 成功案例</a> 
   <ul>
     <li><a href="#AIGC">AIGC: 加速 Stable Diffusion</a></li>
     <li><a href="#生物医药">生物医药: 加速AlphaFold蛋白质结构预测</a></li>
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
 <li><a href="#引用我们">引用我们</a></li>
</ul>

## 为何选择 Colossal-AI
<div align="center">
   <a href="https://youtu.be/KnXSfjqkKN0">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/JamesDemmel_Colossal-AI.png" width="600" />
   </a>

   James Demmel 教授 (加州大学伯克利分校): Colossal-AI 让分布式训练高效、易用、可扩展。
</div>

<p align="right">(<a href="#top">返回顶端</a>)</p>

## 特点

Colossal-AI 为您提供了一系列并行组件。我们的目标是让您的分布式 AI 模型像构建普通的单 GPU 模型一样简单。我们提供的友好工具可以让您在几行代码内快速开始分布式训练和推理。

- 并行化策略
  - 数据并行
  - 流水线并行
  - 1维, [2维](https://arxiv.org/abs/2104.05343), [2.5维](https://arxiv.org/abs/2105.14500), [3维](https://arxiv.org/abs/2105.14450) 张量并行
  - [序列并行](https://arxiv.org/abs/2105.13120)
  - [零冗余优化器 (ZeRO)](https://arxiv.org/abs/1910.02054)
- 异构内存管理
  - [PatrickStar](https://arxiv.org/abs/2108.05818)
- 使用友好
  - 基于参数文件的并行化
- 推理
  - [Energon-AI](https://github.com/hpcaitech/EnergonAI)
- Colossal-AI 成功案例
  - 生物医药: [FastFold](https://github.com/hpcaitech/FastFold) 加速蛋白质结构预测 AlphaFold 训练与推理
<p align="right">(<a href="#top">返回顶端</a>)</p>

## 并行训练样例展示
### ViT
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/ViT.png" width="450" />
</p>

- 14倍批大小和5倍训练速度（张量并行=64）

### GPT-3
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT3-v5.png" width=700/>
</p>

- 释放 50% GPU 资源占用, 或 10.7% 加速

### GPT-2
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT2.png" width=800/>

- 降低11倍 GPU 显存占用，或超线性扩展（张量并行）

<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/(updated)GPT-2.png" width=800>

- 用相同的硬件训练24倍大的模型
- 超3倍的吞吐量 

### BERT
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/BERT.png" width=800/>

- 2倍训练速度，或1.5倍序列长度

### PaLM
- [PaLM-colossalai](https://github.com/hpcaitech/PaLM-colossalai): 可扩展的谷歌 Pathways Language Model ([PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html)) 实现。

### OPT
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/OPT_update.png" width=800/>

- [Open Pretrained Transformer (OPT)](https://github.com/facebookresearch/metaseq), 由Meta发布的1750亿语言模型，由于完全公开了预训练参数权重，因此促进了下游任务和应用部署的发展。
- 加速45%，仅用几行代码以低成本微调OPT。[[样例]](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/language/opt) [[在线推理]](https://service.colossalai.org/opt) 

请访问我们的 [文档](https://www.colossalai.org/) 和 [例程](https://github.com/hpcaitech/ColossalAI-Examples) 以了解详情。


### 推荐系统模型
- [Cached Embedding](https://github.com/hpcaitech/CachedEmbedding), 使用软件Cache实现Embeddings，用更少GPU显存训练更大的模型。


<p align="right">(<a href="#top">返回顶端</a>)</p>

## 单GPU训练样例展示

### GPT-2
<p id="GPT-2-Single" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT2-GPU1.png" width=450/>
</p>

- 用相同的硬件训练20倍大的模型

<p id="GPT-2-NVME" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT2-NVME.png" width=800/>
</p>

- 用相同的硬件训练120倍大的模型 (RTX 3080)

### PaLM
<p id="PaLM-Single" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/PaLM-GPU1.png" width=450/>
</p>

- 用相同的硬件训练34倍大的模型

<p align="right">(<a href="#top">返回顶端</a>)</p>


## 推理 (Energon-AI) 样例展示

<p id="GPT-3-Inference" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference_GPT-3.jpg" width=800/>
</p>

- [Energon-AI](https://github.com/hpcaitech/EnergonAI) ：用相同的硬件推理加速50%

<p id="OPT-Serving" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/OPT_serving.png" width=800/>
</p>

- [OPT推理服务](https://service.colossalai.org/opt): 无需注册，免费体验1750亿参数OPT在线推理服务


<p align="right">(<a href="#top">返回顶端</a>)</p>

## Colossal-AI 成功案例

### AIGC
加速AIGC(AI内容生成)模型，如[Stable Diffusion](https://github.com/CompVis/stable-diffusion)
<p id="diffusion_train" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/diffusion_train.png" width=800/>
</p>

- [Colossal-AI优化Stable Diffusion](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/diffusion): 6.5倍训练加速和预训练成本降低, 微调硬件成本下降约7倍(从RTX3090/4090到RTX3050/2070)

<p id="diffusion_demo" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/diffusion_demo.png" width=800/>
</p>

<p align="right">(<a href="#top">返回顶端</a>)</p>

### 生物医药

加速 [AlphaFold](https://alphafold.ebi.ac.uk/) 蛋白质结构预测

<p id="FastFold" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/FastFold.jpg" width=800/>
</p>

- [FastFold](https://github.com/hpcaitech/FastFold): 加速AlphaFold训练与推理、数据前处理、推理序列长度超过10000残基

<p id="xTrimoMultimer" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/xTrimoMultimer_Table.jpg" width=800/>
</p>

- [xTrimoMultimer](https://github.com/biomap-research/xTrimoMultimer): 11倍加速蛋白质单体与复合物结构预测

<p align="right">(<a href="#top">返回顶端</a>)</p>

## 安装

### 从官方安装

您可以访问我们[下载](https://www.colossalai.org/download)页面来安装Colossal-AI，在这个页面上发布的版本都预编译了CUDA扩展。

### 从源安装

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

<p align="right">(<a href="#top">返回顶端</a>)</p>

## 使用 Docker

### 从DockerHub获取镜像

您可以直接从我们的[DockerHub主页](https://hub.docker.com/r/hpcaitech/colossalai)获取最新的镜像，每一次发布我们都会自动上传最新的镜像。

### 本地构建镜像

运行以下命令从我们提供的 docker 文件中建立 docker 镜像。

> 在Dockerfile里编译Colossal-AI需要有GPU支持，您需要将Nvidia Docker Runtime设置为默认的Runtime。更多信息可以点击[这里](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime)。
> 我们推荐从[项目主页](https://www.colossalai.org)直接下载Colossal-AI.

```bash
cd ColossalAI
docker build -t colossalai ./docker
```

运行以下命令从以交互式启动 docker 镜像.

```bash
docker run -ti --gpus all --rm --ipc=host colossalai bash
```

<p align="right">(<a href="#top">返回顶端</a>)</p>

## 社区
欢迎通过[论坛](https://github.com/hpcaitech/ColossalAI/discussions),
[Slack](https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-z7b26eeb-CBp7jouvu~r0~lcFzX832w),
或[微信](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png "qrcode")加入 Colossal-AI 社区，与我们分享你的建议和问题。


## 做出贡献

欢迎为该项目做出贡献，请参阅[贡献指南](./CONTRIBUTING.md)。

真诚感谢所有贡献者！

<a href="https://github.com/hpcaitech/ColossalAI/graphs/contributors"><img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/contributor_avatar.png" width="800px"></a>

*贡献者头像的展示顺序是随机的。*

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
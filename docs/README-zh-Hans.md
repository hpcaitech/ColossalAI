# Colossal-AI
<div id="top" align="center">

   [![logo](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/colossal-ai_logo_vertical.png)](https://www.colossalai.org/)

   Colossal-AI: 让AI大模型更低成本、方便易用、高效扩展

   <h3> <a href="https://arxiv.org/abs/2110.14883"> 论文 </a> |
   <a href="https://www.colossalai.org/"> 文档 </a> |
   <a href="https://github.com/hpcaitech/ColossalAI/tree/main/examples"> 例程 </a> |
   <a href="https://github.com/hpcaitech/ColossalAI/discussions"> 论坛 </a> |
   <a href="https://medium.com/@hpcaitech"> 博客 </a></h3>

   [![GitHub Repo stars](https://img.shields.io/github/stars/hpcaitech/ColossalAI?style=social)](https://github.com/hpcaitech/ColossalAI/stargazers)
   [![Build](https://github.com/hpcaitech/ColossalAI/actions/workflows/build_on_schedule.yml/badge.svg)](https://github.com/hpcaitech/ColossalAI/actions/workflows/build_on_schedule.yml)
   [![Documentation](https://readthedocs.org/projects/colossalai/badge/?version=latest)](https://colossalai.readthedocs.io/en/latest/?badge=latest)
   [![CodeFactor](https://www.codefactor.io/repository/github/hpcaitech/colossalai/badge)](https://www.codefactor.io/repository/github/hpcaitech/colossalai)
   [![HuggingFace badge](https://img.shields.io/badge/%F0%9F%A4%97HuggingFace-Join-yellow)](https://huggingface.co/hpcai-tech)
   [![slack badge](https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp)](https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-z7b26eeb-CBp7jouvu~r0~lcFzX832w)
   [![WeChat badge](https://img.shields.io/badge/微信-加入-green?logo=wechat&amp)](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png)

   | [English](README.md) | [中文](README-zh-Hans.md) |

</div>

## 新闻
* [2023/03] [ColossalChat: An Open-Source Solution for Cloning ChatGPT With a Complete RLHF Pipeline](https://medium.com/@yangyou_berkeley/colossalchat-an-open-source-solution-for-cloning-chatgpt-with-a-complete-rlhf-pipeline-5edf08fb538b)
* [2023/03] [AWS and Google Fund Colossal-AI with Startup Cloud Programs](https://www.hpc-ai.tech/blog/aws-and-google-fund-colossal-ai-with-startup-cloud-programs)
* [2023/02] [Open Source Solution Replicates ChatGPT Training Process! Ready to go with only 1.6GB GPU Memory](https://www.hpc-ai.tech/blog/colossal-ai-chatgpt)
* [2023/01] [Hardware Savings Up to 46 Times for AIGC and  Automatic Parallelism](https://medium.com/pytorch/latest-colossal-ai-boasts-novel-automatic-parallelism-and-offers-savings-up-to-46x-for-stable-1453b48f3f02)
* [2022/11] [Diffusion Pretraining and Hardware Fine-Tuning Can Be Almost 7X Cheaper](https://www.hpc-ai.tech/blog/diffusion-pretraining-and-hardware-fine-tuning-can-be-almost-7x-cheaper)
* [2022/10] [Use a Laptop to Analyze 90% of Proteins, With a Single-GPU Inference Sequence Exceeding 10,000](https://www.hpc-ai.tech/blog/use-a-laptop-to-analyze-90-of-proteins-with-a-single-gpu-inference-sequence-exceeding)
* [2022/09] [HPC-AI Tech Completes $6 Million Seed and Angel Round Fundraising](https://www.hpc-ai.tech/blog/hpc-ai-tech-completes-6-million-seed-and-angel-round-fundraising-led-by-bluerun-ventures-in-the)


## 目录
<ul>
 <li><a href="#为何选择-Colossal-AI">为何选择 Colossal-AI</a> </li>
 <li><a href="#特点">特点</a> </li>
 <li>
   <a href="#并行训练样例展示">并行训练样例展示</a>
   <ul>
     <li><a href="#GPT-3">GPT-3</a></li>
     <li><a href="#GPT-2">GPT-2</a></li>
     <li><a href="#BERT">BERT</a></li>
     <li><a href="#PaLM">PaLM</a></li>
     <li><a href="#OPT">OPT</a></li>
     <li><a href="#ViT">ViT</a></li>
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
     <li><a href="#BLOOM-Inference">1760亿参数 BLOOM</a></li>
   </ul>
 </li>
<li>
   <a href="#Colossal-AI-in-the-Real-World">Colossal-AI 成功案例</a>
   <ul>
     <li><a href="#ColossalChat">ColossalChat：完整RLHF流程0门槛克隆ChatGPT</a></li>
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
  - [自动并行](https://arxiv.org/abs/2302.02599)
- 异构内存管理
  - [PatrickStar](https://arxiv.org/abs/2108.05818)
- 使用友好
  - 基于参数文件的并行化
- 推理
  - [Energon-AI](https://github.com/hpcaitech/EnergonAI)

<p align="right">(<a href="#top">返回顶端</a>)</p>

## 并行训练样例展示


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
- 加速45%，仅用几行代码以低成本微调OPT。[[样例]](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/opt) [[在线推理]](https://colossalai.org/docs/advanced_tutorials/opt_service)

请访问我们的 [文档](https://www.colossalai.org/) 和 [例程](https://github.com/hpcaitech/ColossalAI/tree/main/examples) 以了解详情。

### ViT
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/ViT.png" width="450" />
</p>

- 14倍批大小和5倍训练速度（张量并行=64）

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
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/BLOOM%20serving.png" width=600/>
</p>

- [OPT推理服务](https://colossalai.org/docs/advanced_tutorials/opt_service): 体验1750亿参数OPT在线推理服务

<p id="BLOOM-Inference" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/BLOOM%20Inference.PNG" width=800/>
</p>

- [BLOOM](https://github.com/hpcaitech/EnergonAI/tree/main/examples/bloom): 降低1760亿参数BLOOM模型部署推理成本超10倍

<p align="right">(<a href="#top">返回顶端</a>)</p>

## Colossal-AI 成功案例
### ColossalChat

<div align="center">
   <a href="https://chat.colossalai.org/">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/Chat-demo.png" width="700" />
   </a>
</div>

[ColossalChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat): 完整RLHF流程0门槛克隆 [ChatGPT](https://openai.com/blog/chatgpt/) [[代码]](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat) [[博客]](https://medium.com/@yangyou_berkeley/colossalchat-an-open-source-solution-for-cloning-chatgpt-with-a-complete-rlhf-pipeline-5edf08fb538b) [[在线样例]](https://chat.colossalai.org)

<p id="ColossalChat_scaling" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chatgpt/ChatGPT%20scaling.png" width=800/>
</p>

- 最高可提升单机训练速度7.73倍，单卡推理速度1.42倍

<p id="ColossalChat-1GPU" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chatgpt/ChatGPT-1GPU.jpg" width=450/>
</p>

- 单卡模型容量最多提升10.3倍
- 最小demo训练流程最低仅需1.62GB显存 (任意消费级GPU)

<p id="ColossalChat-LoRA" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chatgpt/LoRA%20data.jpg" width=600/>
</p>

- 提升单卡的微调模型容量3.7倍
- 同时保持高速运行

<p align="right">(<a href="#top">back to top</a>)</p>

### AIGC
加速AIGC(AI内容生成)模型，如[Stable Diffusion v1](https://github.com/CompVis/stable-diffusion) 和 [Stable Diffusion v2](https://github.com/Stability-AI/stablediffusion)

<p id="diffusion_train" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/Stable%20Diffusion%20v2.png" width=800/>
</p>

- [训练](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/diffusion): 减少5.6倍显存消耗，硬件成本最高降低46倍(从A100到RTX3060)

<p id="diffusion_demo" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/DreamBooth.png" width=800/>
</p>

- [DreamBooth微调](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/dreambooth): 仅需3-5张目标主题图像个性化微调

<p id="inference" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/Stable%20Diffusion%20Inference.jpg" width=800/>
</p>

- [推理](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/diffusion): GPU推理显存消耗降低2.5倍


<p align="right">(<a href="#top">返回顶端</a>)</p>

### 生物医药

加速 [AlphaFold](https://alphafold.ebi.ac.uk/) 蛋白质结构预测

<p id="FastFold" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/FastFold.jpg" width=800/>
</p>

- [FastFold](https://github.com/hpcaitech/FastFold): 加速AlphaFold训练与推理、数据前处理、推理序列长度超过10000残基

<p id="FastFold-Intel" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/data%20preprocessing%20with%20Intel.jpg" width=600/>
</p>

- [FastFold with Intel](https://github.com/hpcaitech/FastFold): 3倍推理加速和39%成本节省

<p id="xTrimoMultimer" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/xTrimoMultimer_Table.jpg" width=800/>
</p>

- [xTrimoMultimer](https://github.com/biomap-research/xTrimoMultimer): 11倍加速蛋白质单体与复合物结构预测

<p align="right">(<a href="#top">返回顶端</a>)</p>

## 安装

环境要求:

- PyTorch >= 1.11 (PyTorch 2.x 正在适配中)
- Python >= 3.7
- CUDA >= 11.0

如果你遇到安装问题，可以向本项目 [反馈](https://github.com/hpcaitech/ColossalAI/issues/new/choose)。


### 从PyPI安装

您可以用下面的命令直接从PyPI上下载并安装Colossal-AI。我们默认不会安装PyTorch扩展包。

```bash
pip install colossalai
```

**注：目前只支持Linux。**

但是，如果你想在安装时就直接构建PyTorch扩展，您可以设置环境变量`CUDA_EXT=1`.

```bash
CUDA_EXT=1 pip install colossalai
```

**否则，PyTorch扩展只会在你实际需要使用他们时在运行时里被构建。**

与此同时，我们也每周定时发布Nightly版本，这能让你提前体验到新的feature和bug fix。你可以通过以下命令安装Nightly版本。

```bash
pip install colossalai-nightly
```

### 从源码安装

> 此文档将与版本库的主分支保持一致。如果您遇到任何问题，欢迎给我们提 issue :)

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# install dependency
pip install -r requirements/requirements.txt

# install colossalai
pip install .
```

我们默认在`pip install`时不安装PyTorch扩展，而是在运行时临时编译，如果你想要提前安装这些扩展的话（在使用融合优化器时会用到），可以使用一下命令。

```shell
CUDA_EXT=1 pip install .
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

参考社区的成功案例，如 [BLOOM](https://bigscience.huggingface.co/) and [Stable Diffusion](https://en.wikipedia.org/wiki/Stable_Diffusion) 等,
无论是个人开发者，还是算力、数据、模型等可能合作方，都欢迎参与参与共建 Colossal-AI 社区，拥抱大模型时代！

您可通过以下方式联系或参与：
1. [留下Star ⭐](https://github.com/hpcaitech/ColossalAI/stargazers) 展现你的喜爱和支持，非常感谢!
2. 发布 [issue](https://github.com/hpcaitech/ColossalAI/issues/new/choose), 或者在GitHub根据[贡献指南](https://github.com/hpcaitech/ColossalAI/blob/main/CONTRIBUTING.md) 提交一个 PR。
3. 发送你的正式合作提案到 contact@hpcaitech.com

真诚感谢所有贡献者！

<a href="https://github.com/hpcaitech/ColossalAI/graphs/contributors"><img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/contributor_avatar.png" width="800px"></a>

*贡献者头像的展示顺序是随机的。*

<p align="right">(<a href="#top">返回顶端</a>)</p>


## CI/CD

我们使用[GitHub Actions](https://github.com/features/actions)来自动化大部分开发以及部署流程。如果想了解这些工作流是如何运行的，请查看这个[文档](.github/workflows/README.md).


## 引用我们

Colossal-AI项目受一些相关的项目启发而成立，一些项目是我们的开发者的科研项目，另一些来自于其他组织的科研工作。我们希望. 我们希望在[参考文献列表](./REFERENCE.md)中列出这些令人称赞的项目，以向开源社区和研究项目致谢。

你可以通过以下格式引用这个项目。

```
@article{bian2021colossal,
  title={Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training},
  author={Bian, Zhengda and Liu, Hongxin and Wang, Boxiang and Huang, Haichen and Li, Yongbin and Wang, Chuanrui and Cui, Fan and You, Yang},
  journal={arXiv preprint arXiv:2110.14883},
  year={2021}
}
```

Colossal-AI 已被 [SC](https://sc22.supercomputing.org/), [AAAI](https://aaai.org/Conferences/AAAI-23/), [PPoPP](https://ppopp23.sigplan.org/), [CVPR](https://cvpr2023.thecvf.com/), [ISC](https://www.isc-hpc.com/)等顶级会议录取为官方教程。

<p align="right">(<a href="#top">返回顶端</a>)</p>

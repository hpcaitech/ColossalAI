# Colossal-AI
<div id="top" align="center">

   [![logo](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/colossal-ai_logo_vertical.png)](https://www.colossalai.org/)

   Colossal-AI: Making large AI models cheaper, faster, and more accessible

   <h3> <a href="https://arxiv.org/abs/2110.14883"> Paper </a> |
   <a href="https://www.colossalai.org/"> Documentation </a> |
   <a href="https://github.com/hpcaitech/ColossalAI/tree/main/examples"> Examples </a> |
   <a href="https://github.com/hpcaitech/ColossalAI/discussions"> Forum </a> |
   <a href="https://medium.com/@hpcaitech"> Blog </a></h3>

   [![GitHub Repo stars](https://img.shields.io/github/stars/hpcaitech/ColossalAI?style=social)](https://github.com/hpcaitech/ColossalAI/stargazers)
   [![Build](https://github.com/hpcaitech/ColossalAI/actions/workflows/build_on_schedule.yml/badge.svg)](https://github.com/hpcaitech/ColossalAI/actions/workflows/build_on_schedule.yml)
   [![Documentation](https://readthedocs.org/projects/colossalai/badge/?version=latest)](https://colossalai.readthedocs.io/en/latest/?badge=latest)
   [![CodeFactor](https://www.codefactor.io/repository/github/hpcaitech/colossalai/badge)](https://www.codefactor.io/repository/github/hpcaitech/colossalai)
   [![HuggingFace badge](https://img.shields.io/badge/%F0%9F%A4%97HuggingFace-Join-yellow)](https://huggingface.co/hpcai-tech)
   [![slack badge](https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp)](https://github.com/hpcaitech/public_assets/tree/main/colossalai/contact/slack)
   [![WeChat badge](https://img.shields.io/badge/微信-加入-green?logo=wechat&amp)](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png)


   | [English](README.md) | [中文](docs/README-zh-Hans.md) |

</div>

## Latest News
* [2023/09] [One Half-Day of Training Using a Few Hundred Dollars Yields Similar Results to Mainstream Large Models, Open-Source and Commercial-Free Domain-Specific Llm Solution](https://www.hpc-ai.tech/blog/one-half-day-of-training-using-a-few-hundred-dollars-yields-similar-results-to-mainstream-large-models-open-source-and-commercial-free-domain-specific-llm-solution)
* [2023/09] [70 Billion Parameter LLaMA2 Model Training Accelerated by 195%](https://www.hpc-ai.tech/blog/70b-llama2-training)
* [2023/07] [HPC-AI Tech Raises 22 Million USD in Series A Funding](https://www.hpc-ai.tech/blog/hpc-ai-tech-raises-22-million-usd-in-series-a-funding-to-fuel-team-expansion-and-business-growth)
* [2023/07] [65B Model Pretraining Accelerated by 38%, Best Practices for Building LLaMA-Like Base Models Open-Source](https://www.hpc-ai.tech/blog/large-model-pretraining)
* [2023/03] [ColossalChat: An Open-Source Solution for Cloning ChatGPT With a Complete RLHF Pipeline](https://medium.com/@yangyou_berkeley/colossalchat-an-open-source-solution-for-cloning-chatgpt-with-a-complete-rlhf-pipeline-5edf08fb538b)
* [2023/03] [Intel and Colossal-AI Partner to Deliver Cost-Efficient Open-Source Solution for Protein Folding Structure Prediction](https://www.hpc-ai.tech/blog/intel-habana)
* [2023/03] [AWS and Google Fund Colossal-AI with Startup Cloud Programs](https://www.hpc-ai.tech/blog/aws-and-google-fund-colossal-ai-with-startup-cloud-programs)
* [2023/02] [Open Source Solution Replicates ChatGPT Training Process! Ready to go with only 1.6GB GPU Memory](https://www.hpc-ai.tech/blog/colossal-ai-chatgpt)
* [2023/01] [Hardware Savings Up to 46 Times for AIGC and  Automatic Parallelism](https://medium.com/pytorch/latest-colossal-ai-boasts-novel-automatic-parallelism-and-offers-savings-up-to-46x-for-stable-1453b48f3f02)

## Table of Contents
<ul>
 <li><a href="#Why-Colossal-AI">Why Colossal-AI</a> </li>
 <li><a href="#Features">Features</a> </li>
 <li>
   <a href="#Colossal-AI-in-the-Real-World">Colossal-AI for Real World Applications</a>
   <ul>
     <li><a href="#Colossal-LLaMA-2">Colossal-LLaMA-2: One Half-Day of Training Using a Few Hundred Dollars Yields Similar Results to Mainstream Large Models, Open-Source and Commercial-Free Domain-Specific Llm Solution</a></li>
     <li><a href="#ColossalChat">ColossalChat: An Open-Source Solution for Cloning ChatGPT With a Complete RLHF Pipeline</a></li>
     <li><a href="#AIGC">AIGC: Acceleration of Stable Diffusion</a></li>
     <li><a href="#Biomedicine">Biomedicine: Acceleration of AlphaFold Protein Structure</a></li>
   </ul>
 </li>
 <li>
   <a href="#Parallel-Training-Demo">Parallel Training Demo</a>
   <ul>
     <li><a href="#LLaMA2">LLaMA 1/2</a></li>
     <li><a href="#GPT-3">GPT-3</a></li>
     <li><a href="#GPT-2">GPT-2</a></li>
     <li><a href="#BERT">BERT</a></li>
     <li><a href="#PaLM">PaLM</a></li>
     <li><a href="#OPT">OPT</a></li>
     <li><a href="#ViT">ViT</a></li>
     <li><a href="#Recommendation-System-Models">Recommendation System Models</a></li>
   </ul>
 </li>
 <li>
   <a href="#Single-GPU-Training-Demo">Single GPU Training Demo</a>
   <ul>
     <li><a href="#GPT-2-Single">GPT-2</a></li>
     <li><a href="#PaLM-Single">PaLM</a></li>
   </ul>
 </li>
 <li>
   <a href="#Inference-Energon-AI-Demo">Inference (Energon-AI) Demo</a>
   <ul>
     <li><a href="#GPT-3-Inference">GPT-3</a></li>
     <li><a href="#OPT-Serving">OPT-175B Online Serving for Text Generation</a></li>
     <li><a href="#BLOOM-Inference">176B BLOOM</a></li>
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
 <li><a href="#Contributing">Contributing</a></li>
 <li><a href="#Cite-Us">Cite Us</a></li>
</ul>

## Why Colossal-AI
<div align="center">
   <a href="https://youtu.be/KnXSfjqkKN0">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/JamesDemmel_Colossal-AI.png" width="600" />
   </a>

   Prof. James Demmel (UC Berkeley): Colossal-AI makes training AI models efficient, easy, and scalable.
</div>

<p align="right">(<a href="#top">back to top</a>)</p>

## Features

Colossal-AI provides a collection of parallel components for you. We aim to support you to write your
distributed deep learning models just like how you write your model on your laptop. We provide user-friendly tools to kickstart
distributed training and inference in a few lines.

- Parallelism strategies
  - Data Parallelism
  - Pipeline Parallelism
  - 1D, [2D](https://arxiv.org/abs/2104.05343), [2.5D](https://arxiv.org/abs/2105.14500), [3D](https://arxiv.org/abs/2105.14450) Tensor Parallelism
  - [Sequence Parallelism](https://arxiv.org/abs/2105.13120)
  - [Zero Redundancy Optimizer (ZeRO)](https://arxiv.org/abs/1910.02054)
  - [Auto-Parallelism](https://arxiv.org/abs/2302.02599)

- Heterogeneous Memory Management
  - [PatrickStar](https://arxiv.org/abs/2108.05818)

- Friendly Usage
  - Parallelism based on the configuration file

- Inference
  - [Energon-AI](https://github.com/hpcaitech/EnergonAI)

<p align="right">(<a href="#top">back to top</a>)</p>

## Colossal-AI in the Real World

### Colossal-LLaMA-2

- One half-day of training using a few hundred dollars yields similar results to mainstream large models, open-source and commercial-free domain-specific LLM solution.
[[code]](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Colossal-LLaMA-2)
[[blog]](https://www.hpc-ai.tech/blog/one-half-day-of-training-using-a-few-hundred-dollars-yields-similar-results-to-mainstream-large-models-open-source-and-commercial-free-domain-specific-llm-solution)
[[HuggingFace model weights]](https://huggingface.co/hpcai-tech/Colossal-LLaMA-2-7b-base)
[[Modelscope model weights]](https://www.modelscope.cn/models/colossalai/Colossal-LLaMA-2-7b-base/summary)

|                                |  Backbone  | Tokens Consumed |  |         MMLU         |     CMMLU     | AGIEval | GAOKAO | CEval  |
| :----------------------------: | :--------: | :-------------: | :------------------: | :-----------: | :-----: | :----: | :----: | :------------------------------: |
|                                |           |        -        |                |        5-shot        |    5-shot     | 5-shot  | 0-shot | 5-shot |
|          Baichuan-7B           |     -      |      1.2T       |             |    42.32 (42.30)     | 44.53 (44.02) |  38.72  | 36.74  | 42.80  |
|       Baichuan-13B-Base        |     -      |      1.4T       |             |    50.51 (51.60)     | 55.73 (55.30) |  47.20  | 51.41  | 53.60  |
|       Baichuan2-7B-Base        |     -      |      2.6T       |             |    46.97 (54.16)     | 57.67 (57.07) |  45.76  | 52.60  | 54.00  |
|       Baichuan2-13B-Base       |     -      |      2.6T       |             |    54.84 (59.17)     | 62.62 (61.97) |  52.08  | 58.25  | 58.10  |
|           ChatGLM-6B           |     -      |      1.0T       |             |    39.67 (40.63)     |   41.17 (-)   |  40.10  | 36.53  | 38.90  |
|          ChatGLM2-6B           |     -      |      1.4T       |             |    44.74 (45.46)     |   49.40 (-)   |  46.36  | 45.49  | 51.70  |
|          InternLM-7B           |     -      |      1.6T       |                |    46.70 (51.00)     |   52.00 (-)   |  44.77  | 61.64  | 52.80  |
|            Qwen-7B             |     -      |      2.2T       |             | 54.29 (56.70) | 56.03 (58.80) |  52.47  | 56.42  | 59.60  |
|                                |            |                 |                 |                      |               |         |        |        |
|           Llama-2-7B           |     -      |      2.0T       |             |    44.47 (45.30)     |   32.97 (-)   |  32.60  | 25.46  |   -    |
| Linly-AI/Chinese-LLaMA-2-7B-hf | Llama-2-7B |      1.0T       |             |        37.43         |     29.92     |  32.00  | 27.57  |   -    |
| wenge-research/yayi-7b-llama2  | Llama-2-7B |        -        |                |        38.56         |     31.52     |  30.99  | 25.95  |   -    |
| ziqingyang/chinese-llama-2-7b  | Llama-2-7B |        -        |                |        33.86         |     34.69     |  34.52  | 25.18  |  34.2  |
| TigerResearch/tigerbot-7b-base | Llama-2-7B |      0.3T       |             |        43.73         |     42.04     |  37.64  | 30.61  |   -    |
|  LinkSoul/Chinese-Llama-2-7b   | Llama-2-7B |        -        |                |        48.41         |     38.31     |  38.45  | 27.72  |   -    |
|       FlagAlpha/Atom-7B        | Llama-2-7B |      0.1T       |             |        49.96         |     41.10     |  39.83  | 33.00  |   -    |
| IDEA-CCNL/Ziya-LLaMA-13B-v1.1  | Llama-13B  |      0.11T      |            |        50.25         |     40.99     |  40.04  | 30.54  |   -    |
|  |  |  |  |  |  |  |  |  |
|    **Colossal-LLaMA-2-7b-base**    | Llama-2-7B |      **0.0085T**      |            |        53.06         |     49.89     |  51.48  | 58.82  |  50.2  |

### ColossalChat

<div align="center">
   <a href="https://www.youtube.com/watch?v=HcTiHzApHm0">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/ColossalChat%20YouTube.png" width="700" />
   </a>
</div>

[ColossalChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat): An open-source solution for cloning [ChatGPT](https://openai.com/blog/chatgpt/) with a complete RLHF pipeline.
[[code]](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat)
[[blog]](https://medium.com/@yangyou_berkeley/colossalchat-an-open-source-solution-for-cloning-chatgpt-with-a-complete-rlhf-pipeline-5edf08fb538b)
[[demo]](https://www.youtube.com/watch?v=HcTiHzApHm0)
[[tutorial]](https://www.youtube.com/watch?v=-qFBZFmOJfg)

<p id="ColossalChat-Speed" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/ColossalChat%20Speed.jpg" width=450/>
</p>

- Up to 10 times faster for RLHF PPO Stage3 Training

<p id="ColossalChat_scaling" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chatgpt/ChatGPT%20scaling.png" width=800/>
</p>

- Up to 7.73 times faster for single server training and 1.42 times faster for single-GPU inference

<p id="ColossalChat-1GPU" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chatgpt/ChatGPT-1GPU.jpg" width=450/>
</p>

- Up to 10.3x growth in model capacity on one GPU
- A mini demo training process requires only 1.62GB of GPU memory (any consumer-grade GPU)

<p id="ColossalChat-LoRA" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chatgpt/LoRA%20data.jpg" width=600/>
</p>

- Increase the capacity of the fine-tuning model by up to 3.7 times on a single GPU
- Keep at a sufficiently high running speed

<p align="right">(<a href="#top">back to top</a>)</p>


### AIGC
Acceleration of AIGC (AI-Generated Content) models such as [Stable Diffusion v1](https://github.com/CompVis/stable-diffusion) and [Stable Diffusion v2](https://github.com/Stability-AI/stablediffusion).
<p id="diffusion_train" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/Stable%20Diffusion%20v2.png" width=800/>
</p>

- [Training](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/diffusion): Reduce Stable Diffusion memory consumption by up to 5.6x and hardware cost by up to 46x (from A100 to RTX3060).

<p id="diffusion_demo" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/DreamBooth.png" width=800/>
</p>

- [DreamBooth Fine-tuning](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/dreambooth): Personalize your model using just 3-5 images of the desired subject.

<p id="inference" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/Stable%20Diffusion%20Inference.jpg" width=800/>
</p>

- [Inference](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/diffusion): Reduce inference GPU memory consumption by 2.5x.


<p align="right">(<a href="#top">back to top</a>)</p>

### Biomedicine
Acceleration of [AlphaFold Protein Structure](https://alphafold.ebi.ac.uk/)

<p id="FastFold" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/FastFold.jpg" width=800/>
</p>

- [FastFold](https://github.com/hpcaitech/FastFold): Accelerating training and inference on GPU Clusters, faster data processing, inference sequence containing more than 10000 residues.

<p id="FastFold-Intel" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/data%20preprocessing%20with%20Intel.jpg" width=600/>
</p>

- [FastFold with Intel](https://github.com/hpcaitech/FastFold): 3x inference acceleration and 39% cost reduce.

<p id="xTrimoMultimer" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/xTrimoMultimer_Table.jpg" width=800/>
</p>

- [xTrimoMultimer](https://github.com/biomap-research/xTrimoMultimer): accelerating structure prediction of protein monomers and multimer by 11x.


<p align="right">(<a href="#top">back to top</a>)</p>

## Parallel Training Demo
### LLaMA2
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/llama2_pretraining.png" width=600/>
</p>

- 70 billion parameter LLaMA2 model training accelerated by 195%
[[code]](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/llama2)
[[blog]](https://www.hpc-ai.tech/blog/70b-llama2-training)

### LLaMA1
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/examples/images/LLaMA_pretraining.png" width=600/>
</p>

- 65-billion-parameter large model pretraining accelerated by 38%
[[code]](https://github.com/hpcaitech/ColossalAI/tree/example/llama/examples/language/llama)
[[blog]](https://www.hpc-ai.tech/blog/large-model-pretraining)

### GPT-3
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT3-v5.png" width=700/>
</p>

- Save 50% GPU resources and 10.7% acceleration

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

### OPT
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/OPT_update.png" width=800/>

- [Open Pretrained Transformer (OPT)](https://github.com/facebookresearch/metaseq), a 175-Billion parameter AI language model released by Meta, which stimulates AI programmers to perform various downstream tasks and application deployments because of public pre-trained model weights.
- 45% speedup fine-tuning OPT at low cost in lines. [[Example]](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/opt) [[Online Serving]](https://colossalai.org/docs/advanced_tutorials/opt_service)

Please visit our [documentation](https://www.colossalai.org/) and [examples](https://github.com/hpcaitech/ColossalAI/tree/main/examples) for more details.

### ViT
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/ViT.png" width="450" />
</p>

- 14x larger batch size, and 5x faster training for Tensor Parallelism = 64

### Recommendation System Models
- [Cached Embedding](https://github.com/hpcaitech/CachedEmbedding), utilize software cache to train larger embedding tables with a smaller GPU memory budget.

<p align="right">(<a href="#top">back to top</a>)</p>

## Single GPU Training Demo

### GPT-2
<p id="GPT-2-Single" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT2-GPU1.png" width=450/>
</p>

- 20x larger model size on the same hardware

<p id="GPT-2-NVME" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT2-NVME.png" width=800/>
</p>

- 120x larger model size on the same hardware (RTX 3080)

### PaLM
<p id="PaLM-Single" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/PaLM-GPU1.png" width=450/>
</p>

- 34x larger model size on the same hardware

<p align="right">(<a href="#top">back to top</a>)</p>


## Inference (Energon-AI) Demo

<p id="GPT-3-Inference" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference_GPT-3.jpg" width=800/>
</p>

- [Energon-AI](https://github.com/hpcaitech/EnergonAI): 50% inference acceleration on the same hardware

<p id="OPT-Serving" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/BLOOM%20serving.png" width=600/>
</p>

- [OPT Serving](https://colossalai.org/docs/advanced_tutorials/opt_service): Try 175-billion-parameter OPT online services

<p id="BLOOM-Inference" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/BLOOM%20Inference.PNG" width=800/>
</p>

- [BLOOM](https://github.com/hpcaitech/EnergonAI/tree/main/examples/bloom): Reduce hardware deployment costs of 176-billion-parameter BLOOM by more than 10 times.

<p align="right">(<a href="#top">back to top</a>)</p>

## Installation

Requirements:
- PyTorch >= 1.11 (PyTorch 2.x in progress)
- Python >= 3.7
- CUDA >= 11.0
- [NVIDIA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) >= 7.0 (V100/RTX20 and higher)
- Linux OS

If you encounter any problem with installation, you may want to raise an [issue](https://github.com/hpcaitech/ColossalAI/issues/new/choose) in this repository.

### Install from PyPI

You can easily install Colossal-AI with the following command. **By default, we do not build PyTorch extensions during installation.**

```bash
pip install colossalai
```

**Note: only Linux is supported for now.**

However, if you want to build the PyTorch extensions during installation, you can set `CUDA_EXT=1`.

```bash
CUDA_EXT=1 pip install colossalai
```

**Otherwise, CUDA kernels will be built during runtime when you actually need them.**

We also keep releasing the nightly version to PyPI every week. This allows you to access the unreleased features and bug fixes in the main branch.
Installation can be made via

```bash
pip install colossalai-nightly
```

### Download From Source

> The version of Colossal-AI will be in line with the main branch of the repository. Feel free to raise an issue if you encounter any problems. :)

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# install colossalai
pip install .
```

By default, we do not compile CUDA/C++ kernels. ColossalAI will build them during runtime.
If you want to install and enable CUDA kernel fusion (compulsory installation when using fused optimizer):

```shell
CUDA_EXT=1 pip install .
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

<p align="right">(<a href="#top">back to top</a>)</p>

## Use Docker

### Pull from DockerHub

You can directly pull the docker image from our [DockerHub page](https://hub.docker.com/r/hpcaitech/colossalai). The image is automatically uploaded upon release.


### Build On Your Own

Run the following command to build a docker image from Dockerfile provided.

> Building Colossal-AI from scratch requires GPU support, you need to use Nvidia Docker Runtime as the default when doing `docker build`. More details can be found [here](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime).
> We recommend you install Colossal-AI from our [project page](https://www.colossalai.org) directly.


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
and [WeChat(微信)](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png "qrcode") to share your suggestions, feedback, and questions with our engineering team.

## Contributing
Referring to the successful attempts of [BLOOM](https://bigscience.huggingface.co/) and [Stable Diffusion](https://en.wikipedia.org/wiki/Stable_Diffusion), any and all developers and partners with computing powers, datasets, models are welcome to join and build the Colossal-AI community, making efforts towards the era of big AI models!

You may contact us or participate in the following ways:
1. [Leaving a Star ⭐](https://github.com/hpcaitech/ColossalAI/stargazers) to show your like and support. Thanks!
2. Posting an [issue](https://github.com/hpcaitech/ColossalAI/issues/new/choose), or submitting a PR on GitHub follow the guideline in [Contributing](https://github.com/hpcaitech/ColossalAI/blob/main/CONTRIBUTING.md)
3. Send your official proposal to email contact@hpcaitech.com

Thanks so much to all of our amazing contributors!

<a href="https://github.com/hpcaitech/ColossalAI/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=hpcaitech/ColossalAI"  width="800px"/>
</a>


<p align="right">(<a href="#top">back to top</a>)</p>


## CI/CD

We leverage the power of [GitHub Actions](https://github.com/features/actions) to automate our development, release and deployment workflows. Please check out this [documentation](.github/workflows/README.md) on how the automated workflows are operated.


## Cite Us

This project is inspired by some related projects (some by our team and some by other organizations). We would like to credit these amazing projects as listed in the [Reference List](./docs/REFERENCE.md).

To cite this project, you can use the following BibTeX citation.

```
@article{bian2021colossal,
  title={Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training},
  author={Bian, Zhengda and Liu, Hongxin and Wang, Boxiang and Huang, Haichen and Li, Yongbin and Wang, Chuanrui and Cui, Fan and You, Yang},
  journal={arXiv preprint arXiv:2110.14883},
  year={2021}
}
```

Colossal-AI has been accepted as official tutorial by top conferences [NeurIPS](https://nips.cc/), [SC](https://sc22.supercomputing.org/), [AAAI](https://aaai.org/Conferences/AAAI-23/),
[PPoPP](https://ppopp23.sigplan.org/), [CVPR](https://cvpr2023.thecvf.com/), [ISC](https://www.isc-hpc.com/), [NVIDIA GTC](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-S51482/) ,etc.

<p align="right">(<a href="#top">back to top</a>)</p>

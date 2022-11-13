# Colossal-AI Tutorial Hands-on

## Introduction

Welcome to the [Colossal-AI](https://github.com/hpcaitech/ColossalAI) tutorial, which has been accepted as official tutorials by top conference [SC](https://sc22.supercomputing.org/), [AAAI](https://aaai.org/Conferences/AAAI-23/), [PPoPP](https://ppopp23.sigplan.org/), etc.


[Colossal-AI](https://github.com/hpcaitech/ColossalAI), a unified deep learning system for the big model era, integrates
many advanced technologies such as multi-dimensional tensor parallelism, sequence parallelism, heterogeneous memory management,
large-scale optimization, adaptive task scheduling, etc. By using Colossal-AI, we could help users to efficiently and
quickly deploy large AI model training and inference, reducing large AI model training budgets and scaling down the labor cost of learning and deployment.

### 🚀 Quick Links

[**Colossal-AI**](https://github.com/hpcaitech/ColossalAI) |
[**Paper**](https://arxiv.org/abs/2110.14883) |
[**Documentation**](https://www.colossalai.org/) |
[**Forum**](https://github.com/hpcaitech/ColossalAI/discussions) |
[**Slack**](https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-z7b26eeb-CBp7jouvu~r0~lcFzX832w)


## Prerequisite

To run this example, you only need to have PyTorch and Colossal-AI installed. A sample script to download the dependencies is given below.

```
# install torch 1.12 with CUDA 11.3
# visit https://pytorch.org/get-started/locally/ to download other versions
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# install latest ColossalAI
# visit https://colossalai.org/download to download corresponding version of Colossal-AI
pip install colossalai==0.1.11+torch1.12cu11.3 -f https://release.colossalai.org
```


## Table of Content

 - Multi-dimensional Parallelism
   - Know the components and sketch of Colossal-AI
   - Step-by-step from PyTorch to Colossal-AI
   - Try data/pipeline parallelism and 1D/2D/2.5D/3D tensor parallelism using a unified model
 - Sequence Parallelism
   - Try sequence parallelism with BERT
   - Combination of data/pipeline/sequence parallelism
   - Faster training and longer sequence length
 - Large Batch Training Optimization
   - Comparison of small/large batch size with SGD/LARS optimizer
   - Acceleration from a larger batch size
 - Auto-Parallelism
   - Parallelism with normal non-distributed training code
   - Model tracing + solution solving + runtime communication inserting all in one auto-parallelism system
   - Try single program, multiple data (SPMD) parallel with auto-parallelism SPMD solver on ResNet50
 - Fine-tuning and Serving for OPT
   - Try pre-trained OPT model weights with Colossal-AI
   - Fine-tuning OPT with limited hardware using ZeRO, Gemini and parallelism
   - Deploy the fine-tuned model to inference service
 - Acceleration of Stable Diffusion
   - Stable Diffusion with Lightning
   - Try Lightning Colossal-AI strategy to optimize memory and accelerate speed

## Prepare Common Dataset

**This tutorial folder aims to let the user to quickly try out the training scripts**. One major task for deep learning is data preparataion. To save time on data preparation, we use `CIFAR10` for most tutorials and synthetic datasets if the dataset required is too large. To make the `CIFAR10` dataset shared across the different examples, it should be downloaded in tutorial root directory with the following command.

```python
python download_cifar10.py
```


## Discussion

Discussion about the [Colossal-AI](https://github.com/hpcaitech/ColossalAI) project is always welcomed! We would love to exchange ideas with the community to better help this project grow.
If you think there is a need to discuss anything, you may jump to our [Slack](https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-z7b26eeb-CBp7jouvu~r0~lcFzX832w).

If you encounter any problem while running these tutorials, you may want to raise an [issue](https://github.com/hpcaitech/ColossalAI/issues/new/choose) in this repository.

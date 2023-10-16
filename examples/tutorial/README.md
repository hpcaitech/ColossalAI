# Colossal-AI Tutorial Hands-on

> This path is an abbreviated tutorial prepared for specific activities and may not be maintained in real time. For use of Colossal-AI, please refer to other [examples](https://github.com/hpcaitech/ColossalAI/tree/main/examples) and [documents](https://www.colossalai.org/).

## Introduction

Welcome to the [Colossal-AI](https://github.com/hpcaitech/ColossalAI) tutorial, which has been accepted as official tutorials by top conference [NeurIPS](https://nips.cc/), [SC](https://sc22.supercomputing.org/), [AAAI](https://aaai.org/Conferences/AAAI-23/),
[PPoPP](https://ppopp23.sigplan.org/), [CVPR](https://cvpr2023.thecvf.com/), [ISC](https://www.isc-hpc.com/), [NVIDIA GTC](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-S51482/) ,etc.


[Colossal-AI](https://github.com/hpcaitech/ColossalAI), a unified deep learning system for the big model era, integrates
many advanced technologies such as multi-dimensional tensor parallelism, sequence parallelism, heterogeneous memory management,
large-scale optimization, adaptive task scheduling, etc. By using Colossal-AI, we could help users to efficiently and
quickly deploy large AI model training and inference, reducing large AI model training budgets and scaling down the labor cost of learning and deployment.

### 🚀 Quick Links

[**Colossal-AI**](https://github.com/hpcaitech/ColossalAI) |
[**Paper**](https://arxiv.org/abs/2110.14883) |
[**Documentation**](https://www.colossalai.org/) |
[**Issue**](https://github.com/hpcaitech/ColossalAI/issues/new/choose) |
[**Slack**](https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-z7b26eeb-CBp7jouvu~r0~lcFzX832w)

## Table of Content

 - Multi-dimensional Parallelism [[code]](https://github.com/hpcaitech/ColossalAI/tree/main/examples/tutorial/hybrid_parallel) [[video]](https://www.youtube.com/watch?v=OwUQKdA2Icc)
 - Sequence Parallelism [[code]](https://github.com/hpcaitech/ColossalAI/tree/main/examples/tutorial/sequence_parallel) [[video]](https://www.youtube.com/watch?v=HLLVKb7Cszs)
 - Large Batch Training Optimization [[code]](https://github.com/hpcaitech/ColossalAI/tree/main/examples/tutorial/large_batch_optimizer) [[video]](https://www.youtube.com/watch?v=9Un0ktxJZbI)
 - Automatic Parallelism [[code]](https://github.com/hpcaitech/ColossalAI/tree/main/examples/tutorial/auto_parallel) [[video]](https://www.youtube.com/watch?v=_-2jlyidxqE)
 - Fine-tuning and Inference for OPT [[code]](https://github.com/hpcaitech/ColossalAI/tree/main/examples/tutorial/opt) [[video]](https://www.youtube.com/watch?v=jbEFNVzl67Y)
 - Optimized AlphaFold [[code]](https://github.com/hpcaitech/ColossalAI/tree/main/examples/tutorial/fastfold) [[video]](https://www.youtube.com/watch?v=-zP13LfJP7w)
 - Optimized Stable Diffusion [[code]](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/diffusion) [[video]](https://www.youtube.com/watch?v=8KHeUjjc-XQ)
 - ColossalChat: Cloning ChatGPT with a Complete RLHF Pipeline
[[code]](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat)
[[blog]](https://medium.com/@yangyou_berkeley/colossalchat-an-open-source-solution-for-cloning-chatgpt-with-a-complete-rlhf-pipeline-5edf08fb538b)
[[demo]](https://www.youtube.com/watch?v=HcTiHzApHm0)
[[video]](https://www.youtube.com/watch?v=-qFBZFmOJfg)

## Discussion

Discussion about the [Colossal-AI](https://github.com/hpcaitech/ColossalAI) project is always welcomed! We would love to exchange ideas with the community to better help this project grow.
If you think there is a need to discuss anything, you may jump to our [Slack](https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-z7b26eeb-CBp7jouvu~r0~lcFzX832w).

If you encounter any problem while running these tutorials, you may want to raise an [issue](https://github.com/hpcaitech/ColossalAI/issues/new/choose) in this repository.

## 🛠️ Setup environment
[[video]](https://www.youtube.com/watch?v=dpMYj974ZIc) You should use `conda` to create a virtual environment, we recommend **python 3.8**, e.g. `conda create -n colossal python=3.8`. This installation commands are for CUDA 11.3, if you have a different version of CUDA, please download PyTorch and Colossal-AI accordingly.
You can refer to the [Installation](https://github.com/hpcaitech/ColossalAI#installation) to set up your environment.

You can run `colossalai check -i` to verify if you have correctly set up your environment 🕹️.
![](https://raw.githubusercontent.com/hpcaitech/public_assets/main/examples/tutorial/colossalai%20check%20-i.png)

If you encounter messages like `please install with cuda_ext`, do let me know as it could be a problem of the distribution wheel. 😥

Then clone the Colossal-AI repository from GitHub.
```bash
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI/examples/tutorial
```

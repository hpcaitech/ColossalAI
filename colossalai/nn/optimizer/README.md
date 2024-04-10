# Colossal-AI Optimization Techniques

## Introduction

Welcome to the large-scale deep learning optimization techniques of [Colossal-AI](https://github.com/hpcaitech/ColossalAI),
which has been accepted as official tutorials by top conference [NeurIPS](https://nips.cc/), [SC](https://sc22.supercomputing.org/), [AAAI](https://aaai.org/Conferences/AAAI-23/),
[PPoPP](https://ppopp23.sigplan.org/), [CVPR](https://cvpr2023.thecvf.com/), [ISC](https://www.isc-hpc.com/), [NVIDIA GTC](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-S51482/) ,etc.


[Colossal-AI](https://github.com/hpcaitech/ColossalAI), a unified deep learning system for the big model era, integrates
many advanced technologies such as multi-dimensional tensor parallelism, sequence parallelism, heterogeneous memory management,
large-scale optimization, adaptive task scheduling, etc. By using Colossal-AI, we could help users to efficiently and
quickly deploy large AI model training and inference, reducing large AI model training budgets and scaling down the labor cost of learning and deployment.

### 🚀 Quick Links

[**Colossal-AI**](https://github.com/hpcaitech/ColossalAI) |
[**Paper**](https://arxiv.org/abs/2110.14883) |
[**Documentation**](https://www.colossalai.org/) |
[**Forum**](https://github.com/hpcaitech/ColossalAI/discussions) |
[**Slack**](https://github.com/hpcaitech/public_assets/tree/main/colossalai/contact/slack)


## Table of Content

Large transformer models display promising performance on a wide spectrum of AI applications.
Both academia and industry are scaling DL training on larger clusters. However, degrading generalization performance, non-negligible communication overhead, and increasing model size prevent DL researchers and engineers from exploring large-scale AI models.

We aim to provide a clear sketch of the optimizations for large-scale deep learning with regard to model accuracy and model efficiency.
One way to achieve the goal of maintaining or improving the model accuracy in the large-scale setting while maintaining compute efficiency is to design algorithms that
are less communication and memory hungry. Notably, they are not mutually exclusive but can
be optimized jointly to further speed up training.

1. Model Accuracy
    - Gradient Descent Optimization
      - Gradient Descent Variants
      - Momentum
      - Adaptive Gradient
    - Large Batch Training Optimization
      - LARS
      - LAMB
      - Generalization Gap
    - Second-Order Optimization
      - Hessian-Free
      - K-FAC
      - Shampoo

2. Model Accuracy
    - Communication Efficiency
      - Reduce Volume of Comm.
      - Reduce Frequency of Comm.
    - Memory Efficiency
      - Mix-Precision Training
      - Memory-Efficient Methods, e.g. ZeRO, Gemini, etc.

Some of the above are still under development. **If you wish to make a contribution to this repository, please read the `Contributing` section below.**

## Discussion

Discussion about the Colossal-AI project is always welcomed! We would love to exchange ideas with the community to better help this project grow.
If you think there is a need to discuss anything, you may jump to our [Slack](https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-z7b26eeb-CBp7jouvu~r0~lcFzX832w).

If you encounter any problem while running these optimizers, you may want to raise an issue in this repository.

## Contributing

This project welcomes constructive ideas and implementations from the community.

### Update an Optimizer

If you find that an optimizer is broken (not working) or not user-friendly, you may put up a pull request to this repository and update this optimizer.

### Add a New Optimizer

If you wish to add an optimizer for a specific application, please follow the steps below.

1. create the new optimizer file in the current folder
2. Prepare the corresponding example files in the [Examples](https://github.com/hpcaitech/ColossalAI-Examples) repository to prove effectiveness of the new optimizer
3. Prepare a detailed readme on environment setup, dataset preparation, code execution, etc. in your example folder
4. Update the table of content (last section above) in this readme file


If your PR is accepted, we may invite you to put up a tutorial or blog in [ColossalAI Documentation](https://colossalai.org/).

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


## Optimizer

A series of optimizers have been optimized and integrated.

### Distributed Adafactor

Distributed Adafactor is an optimiser that supports hybrid optimisation, including 1D tensor parallelism as well as ZerO. It makes full use of computational resources through reasonable task parallelism, improves training efficiency and speed, and reduces space pressure on single card storage. It has a wide range of applications and currently supports a range of Transformer based models, see [tests.kit.model_zoo](https://github.com/hpcaitech/ColossalAI/tree/main/tests/kit/model_zoo) for details.  

### Distributed Adafactor API

### Performance
|            Version              |    iter    | Float Percision |      Device Nums     | weight shape  | Avg runtime(ms)  | Avg Speed Up Rate | Best Speed Up Rate  |
| :-----------------------------: | :--------: | :-------------: | :------------------: | :-----------: | :--------------: | :-----------------: | :---------------: |
|           AdaFactor             |     50     |     float32     |          2           | [4096 , 4096] |        0.58      |           -         |          -        |
|    DistAdaFactor(Col Parallel)  |     50     |     float32     |          2           | [4096 , 4096] |        0.41      |         1.39        |        56.91      |
|    DistAdaFactor(Col Parallel)  |     50     |     float32     |          2           | [4096 , 4096] |        0.61      |         0.96        |        18.69      |
|           AdaFactor             |     50     |     float16     |          2           | [4096 , 4096] |        0.72      |           -         |          -        |
|    DistAdaFactor(Col Parallel)  |     50     |     float16     |          2           | [4096 , 4096] |        0.54      |         1.33        |        26.03      |
|    DistAdaFactor(Row Parallel)  |     50     |     float16     |          2           | [4096 , 4096] |        0.67      |         1.08        |        20.55      |
|           AdaFactor             |     50     |     bfloat16    |          2           | [4096 , 4096] |        0.72      |           -         |          -        |
|    DistAdaFactor(Col Parallel)  |     50     |     bfloat16    |          2           | [4096 , 4096] |        0.55      |         1.31        |        26.11      |
|    DistAdaFactor(Row Parallel)  |     50     |     bfloat16    |          2           | [4096 , 4096] |        0.67      |         1.07        |        21.86      |
| :-----------------------------: | :--------: | :-------------: | :------------------: | :-----------: | :--------------: | :-----------------: | :---------------: |
|           AdaFactor             |     50     |     float32     |          4           | [4096 , 4096] |        0.57      |           -         |          -        |
|    DistAdaFactor(Col Parallel)  |     50     |     float32     |          4           | [4096 , 4096] |        0.38      |         1.48        |        53.99      |
|    DistAdaFactor(Col Parallel)  |     50     |     float32     |          4           | [4096 , 4096] |        0.60      |         0.95        |        16.53      |
|           AdaFactor             |     50     |     float16     |          4           | [4096 , 4096] |        0.70      |           -         |          -        |
|    DistAdaFactor(Col Parallel)  |     50     |     float16     |          4           | [4096 , 4096] |        0.50      |         1.44        |        21.98      |
|    DistAdaFactor(Row Parallel)  |     50     |     float16     |          4           | [4096 , 4096] |        0.64      |         1.12        |        15.35      |
|           AdaFactor             |     50     |     bfloat16    |          4           | [4096 , 4096] |        0.72      |           -         |          -        |
|    DistAdaFactor(Col Parallel)  |     50     |     bfloat16    |          4           | [4096 , 4096] |        0.56      |         1.29        |        25.63      |
|    DistAdaFactor(Row Parallel)  |     50     |     bfloat16    |          4           | [4096 , 4096] |        0.71      |         1.09        |        21.52      |
| :-----------------------------: | :--------: | :-------------: | :------------------: | :-----------: | :--------------: | :-----------------: | :---------------: |
|           AdaFactor             |     50     |     float32     |          8           | [4096 , 4096] |        0.56      |           -         |          -        |
|    DistAdaFactor(Col Parallel)  |     50     |     float32     |          8           | [4096 , 4096] |        0.38      |         1.50        |        54.51      |
|    DistAdaFactor(Col Parallel)  |     50     |     float32     |          8           | [4096 , 4096] |        0.91      |         0.67        |        15.68      |
|           AdaFactor             |     50     |     float16     |          8           | [4096 , 4096] |        0.74      |           -         |          -        |
|    DistAdaFactor(Col Parallel)  |     50     |     float16     |          8           | [4096 , 4096] |        0.84      |         0.87        |         9.21      |
|    DistAdaFactor(Row Parallel)  |     50     |     float16     |          8           | [4096 , 4096] |        1.03      |         0.75        |        16.12      |
|           AdaFactor             |     50     |     bfloat16    |          8           | [4096 , 4096] |        0.71      |           -         |          -        |
|    DistAdaFactor(Col Parallel)  |     50     |     bfloat16    |          8           | [4096 , 4096] |        0.54      |         1.31        |        27.28      |
|    DistAdaFactor(Row Parallel)  |     50     |     bfloat16    |          8           | [4096 , 4096] |        0.73      |         1.03        |        25.01      |

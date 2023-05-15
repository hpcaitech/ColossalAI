# Overview

Automatic Mixed Precision training provides significant arithmetic speedup by performing operations in half precision, and offers data transfer speedup by requiring less memory bandwidth. It also allocates less memory, enabling us to train larger models or train with larger batch size. In this example, we use one GPU to reproduce the training of Vision Transformer (ViT) on Caltech101 using colossalai.

You may refer to [our documentation on mixed precision training](https://colossalai.org/tutorials/features/mixed_precision_training) for more details.

> ⚠️ This example is only for demo purpose, no guarantee on the convergence performance

# Prerequisite

```shell
pip install timm scipy titans
```

# How to run

You can execute the following command with 4 GPUs by replacing `config-name` with the actual config file name.
You may change `--nproc_per_node` to the number of GPUs you have on your machine.
The dataset will be downloaded to `./data` by default. If you wish to download it somewhere else, you can run `export DATA=/you/path` in terminal.

```shell
# run with 4 GPUs with booster
colossalai run --nproc_per_node 4 train.py --config ./config/<config-name>.py
```

# Experiments
In order to let everyone have a more intuitive feeling about amp, we use several amp methods to pretrain VIT-Base/16 on ImageNet-1K. The experimental results aims to prove that amp's efficiency in reducing memory and improving efficiency, so that hyperparameters such as learning rate may not be optimal.

|                  | RAM/GB | Iteration/s | throughput (batch/s) |
| ---------------- | ------ | ----------- | -------------------- |
| FP32 training    | 27.2   | 2.95        | 377.6                |
| AMP_TYPE.TORCH   | 20.5   | 3.25        | 416.0                |
| AMP_TYPE.NAIVE   | 17.0   | 3.53        | 451.8                |
| AMP_TYPE.APEX O1 | 20.2   | 3.07        | 393.0                |

As can be seen from the above table, the automixed precision training can reduces the RAM by 37.5% in the best cases, while increasing the throughput by 19.6%. Since the AMP reduces the memory cost for training models, we can further try enabling larger minibatches, which leads to larger throughput.


We also use the example code in this repo to train ViT-Base/16 on caltech101 dataset. The results are as follows:

|                  | RAM/GB | Iteration/s | throughput (batch/s) |
| ---------------- | ------ | ----------- | -------------------- |
| FP32 training    | 25.0   | 0.84        | 107.5                |
| AMP_TYPE.TORCH   | 19.0   | 0.91        | 116.5                |
| AMP_TYPE.NAIVE   | 14.8   | 0.93        | 119.0                |
| AMP_TYPE.APEX O1 | 17.9   | 0.90        | 115.2                |

We observed a significant reduction in memory usage. The amp methods also slightly outperforms the full precision training in efficiency. However, the throughput of AMP training is not as well performed as in the last test(ImageNet-1K), which may be because the dataloader has become the bottleneck. It is very likely that most of the time is spent in reading data, and there is still a large computational advantage. You can add a timer to check the forward/backward time.


# Details
`config.py`
This is a [configuration file](features/amp_with_booster/config/config.py) that defines hyperparameters and training scheme. The config content can be accessed through `gpc.config` in the program. By tuning configuration, this example can be quickly deployed to a single server with several GPUs or to a large cluster with lots of nodes and GPUs.


`train.py`
The script to start training with Colossal-AI.

# Command Line Tool

Author: Shenggui Li

**Prerequisite:**
- [Distributed Training](../concepts/distributed_training.md)
- [Colossal-AI Overview](../concepts/colossalai_overview.md)

## Introduction

Colossal-AI provides command-line utilities for the user.
The current command line tools support the following features.

- verify Colossal-AI build
- launch distributed jobs
- tensor parallel micro-benchmarking

## Check Installation

To verify whether your Colossal-AI is built correctly, you can use the command `colossalai check -i`.
This command will inform you information regarding the version compatibility and cuda extension.

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/05/04/KJmcVknyPHpBofa.png"/>
<figcaption>Check Installation Demo</figcaption>
</figure>

## Launcher

To launch distributed jobs on single or multiple nodes, the command `colossalai run` can be used for process launching.
You may refer to [Launch Colossal-AI](./launch_colossalai.md) for more details.

## Tensor Parallel Micro-Benchmarking

As Colossal-AI provides an array of tensor parallelism methods, it is not intuitive to choose one for your hardware and
model. Therefore, we provide a simple benchmarking to evaluate the performance of various tensor parallelisms on your system.
This benchmarking is run on a simple MLP model where the input data is of the shape `(batch_size, seq_length, hidden_size)`.
Based on the number of GPUs, the CLI will look for all possible tensor parallel configurations and display the benchmarking results.
You can customize the benchmarking configurations by checking out `colossalai benchmark --help`.

```shell
# run on 4 GPUs
colossalai benchmark --gpus 4

# run on 8 GPUs
colossalai benchmark --gpus 8
```

:::caution

Only single-node benchmarking is supported currently.

:::

# Configure Parallelization

Author: Shenggui Li, Siqi Mai

> ⚠️ The information on this page is outdated and will be deprecated. Please check [Booster Plugins](../basics/booster_plugins.md) for more information.

**Prerequisite:**
- [Distributed Training](../concepts/distributed_training.md)
- [Paradigms of Parallelism](../concepts/paradigms_of_parallelism.md)
- [Define Your Configuration](./define_your_config.md)


## Introduction

We support multiple parallelization in Colossal-AI. Hybrid parallelism in our codebase refers to namely the combination
of data parallelism, pipeline parallelism and tensor parallelism (1D, 2D, 2.5D, 3D).

Each parallelism requires different network topology and thus initialize different process groups.
You can initialize the corresponding process group by setting `parallel` in the config file.
The configuration for `parallel` must obey the following format. Data parallel size will be
inferred automatically based on your inputs to pipeline parallelism and tensor parallelism.
`colossalai.launch` will initialize these distributed process groups automatically based on your configuration.

Some sample configurations are shown below:

```python
# sampler format
parallel = dict(
    pipeline=dict("size": int),
    tensor=dict("size": int, "mode": '1d' or '2d' or '2.5d' or '3d', "kwargs": Any)
)

# this is ok
parallel = dict(
    pipeline=dict(size=2),
    tensor=dict(size=4, mode='2d')
)

# this is ok
parallel = dict(
    pipeline=2,
    tensor=dict(size=4, mode='2d')
)

# this is not ok
# as you need to specify the mode for tensor parallelism
parallel = dict(
    pipeline=2,
    tensor=4
)

# this is ok as well as tensor will be default to size 1
# and mode None
parallel = dict(
    pipeline=2
)

# this is ok as well as pipeline will default to size 1
parallel = dict(
    tensor=dict(size=4, mode='2d')
)

```

The key name `size` refers to the parallel size of the parallelism dimension. For example, pipeline size 2 means there
will be 2 pipeline stages. The key name `mode` in tensor parallel config means the corresponding tensor parallelism
will be initialized.

**You can choose to not have 'parallel' in your configuration and both pipeline and tensor will default to size 1.**

**Total number of GPUs must be equal to `data parallel size * tensor parallel size * pipeline parallel size`**

## Data Parallel

Data parallel is the most common way to distribute your training task by splitting data into several shards and train on
a single shard on each device. The configuration for data parallel is detected automatically and set for you. You do not
have to explicitly set them in your configurations. There are two ways to handle the all-reduce in data parallel in Colossal-AI.

1. If you specify gradient handlers, gradients will be all-reduced according to the gradient handlers
2. Otherwise, PyTorch DistributedDataParallel will be used

In most cases, you will be using the second mode unless you have complex handling of the gradients.

## 1D, 2D, 2.5D and 3D Parallel

To enable hybrid parallelism, we provide an array of tensor parallelism. We provide the list of papers which match each
tensor parallel method. These parallel modes need to work with the distributed layers provided by Colossal-AI.

- 1D: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)

- 2D: [An Efficient 2D Method for Training Super-Large Deep Learning Models](https://arxiv.org/abs/2104.05343)
  2D parallel relies on the SUMMA matrix multiplication algorithm and splits the input data, model weights and layer
  outputs along two different dimensions. The tensor chunks are distributed over a 2D mesh of `P = N^2` devices where
  `N` is the number of tensor chunks in a single dimension.

- 2.5D: [2.5-dimensional distributed model training](https://arxiv.org/abs/2105.14500)
  Inspired by the 2.5D matrix multiplication algorithm, 2.5D parallel introduces a novel tensor parallelism which
  further parallelizes 2D tensor parallelism. An amount of `P = N^2 ∗ d` processors are arranged into `d` layers, where
  each layer performs matrix multiplication operations independently with a dimension `N`.

- 3D: [Maximizing Parallelism in Distributed Training for Huge Neural Networks](https://arxiv.org/abs/2105.14450)
  We also introduce a 3D tensor parallelism that parallelizes neural networks on a 3D processor cube. This method
  achieves the optimal, `O(P^{1/3})` communication overhead on $P$ processors, while both computation and memory usage
  are evenly distributed through optimized load balancing of parameters as well as activations.

```python
# 1D parallel
parallel = dict(
    tensor=dict(size=4, mode='1d')
)

# 2D parallel
parallel = dict(
    tensor=dict(size=4, mode='2d')
)

# 2.5D parallel
parallel = dict(
    tensor=dict(size=8, mode='2.5d', depth=2)
)

# 3D parallel
parallel = dict(
    tensor=dict(size=8, mode='3d')
)
```

Once you specify the tensor parallel mode in your configuration, you can proceed to use its corresponding distributed
operator. For example, if you mode is '2d', you can use `colossalai.nn.Linear2D` in you model construction.


## Pipeline Parallel

Pipeline parallelism is to split the model into several partitions by layer. For example, let's assume we have a simple
model which consists of two linear layer. We have two GPUs, and we can allocate the first linear layer to the first GPU
and the second layer to the second GPU.

You can set the number of pipeline stages in your configuration file. When pipeline size is larger than 1, Colossal-AI
will automatically creates the pipeline schedule which defines the forward and backward step.

```python
parallel = dict(
    pipeline=dict(size=4), # number of pipeline stages
)
```

## Sequence Parallel

Sequence parallel is to support long-sequence modelling such as document-level text understanding and medical imaging.
This method is proposed in [Sequence Parallelism: Making 4D Parallelism Possible](https://arxiv.org/abs/2105.13120).
You can use specify the mode to be `sequence` to initialize its process group.


```python
parallel = dict(
    tensor=dict(size=4, mode='sequence')
)
```

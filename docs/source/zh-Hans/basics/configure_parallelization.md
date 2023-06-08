# 并行配置

作者: Shenggui Li, Siqi Mai

> ⚠️ 此页面上的信息已经过时并将被废弃。请在[Booster插件](../basics/booster_plugins.md)页面查阅更新。

**预备知识:**
- [分布式训练](../concepts/distributed_training.md)
- [并行技术](../concepts/paradigms_of_parallelism.md)
- [构建配置文件](./define_your_config.md)


## 简介

我们在 Colossal-AI 中支持多种并行技术。代码库中的混合并行是指您可以轻松地结合数据并行、流水线并行和张量并行（1D、2D、2.5D、3D）的优势共同来进行并行训练。

每种并行方式需要不同的网络拓扑结构，因此要初始化不同的进程组。您可以通过在配置文件中设置 `parallel` 来初始化相应的进程组。 `parallel` 的配置必须遵从以下格式。数据并行度的大小将被根据您对流水线并行和张量并行的输入自动推断。`colossalai.launch` 将根据您的配置自动初始化这些分布式进程组。

我们为您提供了一些配置的例子以供参考。

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

关键字 `size` 指的是并行维度的并行大小。 例如，流水线大小为2意味着有
将有2个流水线阶段。张量并行配置中的关键字 `mode` 意味着相应的张量并行技术
将被初始化，如1D、2D、2.5D、3D。

**您也可以选择不在您的配置中使用 "并行"，此时流水线和张量的并行度都将默认为大小1。**

**GPU的总数量必须等于` 数据并行大小 x 张量并行大小 x 流水线并行大小` 。**

## 数据并行

数据并行是最常见的分布式训练方式。它将数据分割成几个碎片分别在每个设备上进行训练。数据并行的配置会自动检测并为您设置。您不需要在您的配置中明确地设置它们。在Colossal-AI 中，有两种方法来处理数据并行的 all-reduce。

1. 如果您设置了梯度handler，梯度handler将会all-reduce梯度。
2. 若没有指定相应的配置，Colossal-AI 将会使用 PyTorch 的 DistributedDataParallel。

在大多数情况下，若您对梯度没有复杂的处理的需求，您将会使用第二种模式。

## 1D, 2D, 2.5D 和 3D 并行

为了实现混合并行，我们提供了一系列张量并行方法。您可以阅读相应的学术论文进行深入的了解。这些并行模式需要和 Colossal-AI 提供的分布式层一同工作。

- 1D: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)

- 2D: [An Efficient 2D Method for Training Super-Large Deep Learning Models](https://arxiv.org/abs/2104.05343)
  2D 并行基于 SUMMA 矩阵乘法，它将输入数据、模型权重和层输出切分成两个不同的维度。 这些张量块分布在 `P = N^2` 设备的二维网格上，其中 `N` 是单一维度上张量块的数量。

- 2.5D: [2.5-dimensional distributed model training](https://arxiv.org/abs/2105.14500)
  在 2.5D 矩阵乘法的启发下，2.5D 并行引入了一种新的张量并行，进一步将2D张量并行化。其中，`P = N^2 ∗ d` 个处理器被分配到 `d` 层， 每层独立进行矩阵乘法运算，维度为 `N`。

- 3D: [Maximizing Parallelism in Distributed Training for Huge Neural Networks](https://arxiv.org/abs/2105.14450)
  我们还介绍了一种 3D 张量并行方法，在三维处理器立方体上并行化神经网络。这种方法在数量为 `P` 的处理器上实现了最佳的 `O(P^{1/3})` 通信开销，而计算和内存的使用都是通过优化的参数和激活的负载平衡来实现的。同时，通过优化参数和 activations 的负载平衡，计算和内存的使用都是均匀分布的。

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

当您在配置中指定了张量并行模式，您就可以使用其相应的分布式算子。例如，若您设置模式为 `2d`，那么在模型构建中就能使用 `colossalai.nn.Linear2D` 了。


## 流水线并行

流水线并行是将模型按层分成几个部分。例如，假设我们有一个简单的模型，它由两个线性层组成。我们有两个 GPU，我们可以将第一个线性层分配给第一个 GPU 而第二层则分配给第二个 GPU。

您可以在您的配置文件中设置流水线并行度的大小。当流水线并行度大于1，Colossal-AI 将会自动地创建流水线并行的 schedule，这将会为您定义好模型训练的 `forward` 和 `backward`。

```python
parallel = dict(
    pipeline=dict(size=4), # number of pipeline stages
)
```

## 序列并行

针对处理大图片、视频、长文本、长时间医疗监控等数据的需要，Colossal-AI 还提供了序列并行的方法。该方法是在论文[Sequence Parallelism: Making 4D Parallelism Possible](https://arxiv.org/abs/2105.13120)中提出的。您可以指定模式为 `sequence` 来初始化进程组。


```python
parallel = dict(
    tensor=dict(size=4, mode='sequence')
)
```

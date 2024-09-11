# 并行技术

作者: Shenggui Li, Siqi Mai

## 简介

随着深度学习的发展，对并行训练的需求越来越大。这是因为模型和数据集越来越大，如果我们坚持使用单 GPU 训练，训练过程的等待将会成为一场噩梦。在本节中，我们将对现有的并行训练方法进行简要介绍。如果您想对这篇文章进行补充，欢迎在[GitHub论坛](https://github.com/hpcaitech/ColossalAI/discussions)上进行讨论。

## 数据并行

数据并行是最常见的并行形式，因为它很简单。在数据并行训练中，数据集被分割成几个碎片，每个碎片被分配到一个设备上。这相当于沿批次维度对训练过程进行并行化。每个设备将持有一个完整的模型副本，并在分配的数据集碎片上进行训练。在反向传播之后，模型的梯度将被全部减少，以便在不同设备上的模型参数能够保持同步。

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/WSAensMqjwHdOlR.png"/>
<figcaption>数据并行</figcaption>
</figure>

## 模型并行

在数据并行训练中，一个明显的特点是每个 GPU 持有整个模型权重的副本。这就带来了冗余问题。另一种并行模式是模型并行，即模型被分割并分布在一个设备阵列上。通常有两种类型的并行：张量并行和流水线并行。张量并行是在一个操作中进行并行计算，如矩阵-矩阵乘法。流水线并行是在各层之间进行并行计算。因此，从另一个角度来看，张量并行可以被看作是层内并行，流水线并行可以被看作是层间并行。

### 张量并行

张量并行训练是将一个张量沿特定维度分成 `N` 块，每个设备只持有整个张量的 `1/N`，同时不影响计算图的正确性。这需要额外的通信来确保结果的正确性。

以一般的矩阵乘法为例，假设我们有 `C = AB`。我们可以将B沿着列分割成 `[B0 B1 B2 ... Bn]`，每个设备持有一列。然后我们将 `A` 与每个设备上 `B` 中的每一列相乘，我们将得到 `[AB0 AB1 AB2 ... ABn]` 。此刻，每个设备仍然持有一部分的结果，例如，设备(rank=0)持有 `AB0`。为了确保结果的正确性，我们需要收集全部的结果，并沿列维串联张量。通过这种方式，我们能够将张量分布在设备上，同时确保计算流程保持正确。

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/2ZwyPDvXANW4tMG.png"/>
<figcaption>张量并行</figcaption>
</figure>

在 Colossal-AI 中，我们提供了一系列的张量并行方法，即 1D、2D、2.5D 和 3D 张量并行。我们将在`高级教程`中详细讨论它们。


相关文章:
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [An Efficient 2D Method for Training Super-Large Deep Learning Models](https://arxiv.org/abs/2104.05343)
- [2.5-dimensional distributed model training](https://arxiv.org/abs/2105.14500)
- [Maximizing Parallelism in Distributed Training for Huge Neural Networks](https://arxiv.org/abs/2105.14450)

### 流水线并行

流水线并行一般来说很容易理解。请您回忆一下您的计算机结构课程，这确实存在于 CPU 设计中。

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/at3eDv7kKBusxbd.png"/>
<figcaption>流水线并行</figcaption>
</figure>

流水线并行的核心思想是，模型按层分割成若干块，每块都交给一个设备。在前向传递过程中，每个设备将中间的激活传递给下一个阶段。在后向传递过程中，每个设备将输入张量的梯度传回给前一个流水线阶段。这允许设备同时进行计算，并增加了训练的吞吐量。流水线并行训练的一个缺点是，会有一些设备参与计算的冒泡时间，导致计算资源的浪费。

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/sDNq51PS3Gxbw7F.png"/>
<figcaption>Source: <a href="https://arxiv.org/abs/1811.06965">GPipe</a></figcaption>
</figure>

相关文章:
- [PipeDream: Fast and Efficient Pipeline Parallel DNN Training](https://arxiv.org/abs/1806.03377)
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines](https://arxiv.org/abs/2107.06925)

### 序列并行
序列并行是一种对于序列维度进行切分的并行策略，它是训练长文本序列的有效方法。现成熟的序列并行方法包括megatron提出的序列并行，DeepSpeed-Ulysses序列并行和ring-attention序列并行等。
#### megatron sp:

该序列并行方法是在张量并行的基础上实现的序列并行，模型并行的每个gpu上，样本独立且重复的，对于非线性运算的部分如layernorm等无法使用张量并行的模块，可以在序列维度将样本数据切分为多个部分，每个gpu计算部分数据，然后在计算attention及mlp等线性部分使用张量并行策略，需要将activation汇总，这样可以在模型进行切分的情况下进一步减少activation的内存占用，需要注意的是该序列并行方法只能与张量并行一起使用。

#### DeepSpeed-Ulysses:

序列并行通过在序列维度上分割样本并利用all-to-all通信操作，使每个GPU接收完整序列但仅计算注意力头的非重叠子集，从而实现序列并行。该并行方法具有完全通用的attention，可支持密集和稀疏的注意力。
alltoall是一个全交换操作，相当于分布式转置的操作，在attention计算之前，将样本沿序列维度进行切分，每个设备只有N/P的序列长度，然而使用alltoall后，qkv的子部分shape变为[N, d/p]，在计算attention时仍考虑了整体的序列。
#### ring attention：

ring attention思路类似于flash attention，每个GPU只计算一个局部的attention，最后将所有的attention块结果进行归约计算出总的attention。在Ring Attention中，输入序列被沿着序列维度切分为多个块，每个块由不同的GPU或处理器负责处理，Ring Attention采用了一种称为“环形通信”的策略，通过跨卡的p2p通信相互传递kv子块来实现迭代计算，可以实现多卡的超长文本。在这种策略下，每个处理器只与它的前一个和后一个处理器交换信息，形成一个环形网络。通过这种方式，中间结果可以在处理器之间高效传递，而无需全局同步，减少了通信开销。

相关论文：
[Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/pdf/2205.05198)
[DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509)
[Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/pdf/2310.01889)


## 优化器相关的并行

另一种并行方法和优化器相关，目前这种并行最流行的方法是 `ZeRO`，即[零冗余优化器](https://arxiv.org/abs/1910.02054)。 ZeRO 在三个层面上工作，以消除内存冗余（ZeRO需要进行fp16训练）。

- Level 1: 优化器状态在各进程中被划分。
- Level 2: 用于更新模型权重的32位梯度也被划分，因此每个进程只存储与其优化器状态划分相对应的梯度。
- Level 3: 16位模型参数在各进程中被划分。

相关文章:
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)


## 异构系统的并行

上述方法通常需要大量的 GPU 来训练一个大型模型。然而，人们常常忽略的是，与 GPU 相比，CPU 的内存要大得多。在一个典型的服务器上，CPU 可以轻松拥有几百GB的内存，而每个 GPU 通常只有16或32GB的内存。这促使人们思考为什么 CPU 内存没有被用于分布式训练。

最近的进展是依靠 CPU 甚至是 NVMe 磁盘来训练大型模型。主要的想法是，在不使用张量时，将其卸载回 CPU 内存或 NVMe 磁盘。通过使用异构系统架构，有可能在一台机器上容纳一个巨大的模型。

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/qLHD5lk97hXQdbv.png"/>
<figcaption>异构系统</figcaption>
</figure>

相关文章:
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)
- [PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management](https://arxiv.org/abs/2108.05818)
<!-- doc-test-command: echo  -->

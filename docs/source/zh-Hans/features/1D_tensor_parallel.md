# 1D 张量并行

作者: Zhengda Bian, Yongbin Li


**示例代码**
- [Tensor Parallelism with Shardformer](https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/shardformer/examples)

**相关论文**
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://deepakn94.github.io/assets/papers/megatron-sc21.pdf)

## 引言

张量并行将模型参数划分到多个设备上，以减少内存负荷。
[Megatron-LM](https://deepakn94.github.io/assets/papers/megatron-sc21.pdf) 介绍了一种高效的一维张量并行化实现。

让我们以一个线性层为例，它包括一个 GEMM $Y = XA$。 给定2个处理器，我们把列 $A$ 划分为 $[A_1 ~ A_2]$, 并在每个处理器上计算 $Y_i = XA_i$ , 然后形成 $[Y_1 ~ Y_2] = [XA_1 ~ XA_2]$. 这被称为列并行方式。

当第二个线性层 $Z=YB$ 跟随上述列并行层的时候, 我们把 $B$ 划分为
$$
\left[\begin{matrix} B_1 \\ B_2 \end{matrix} \right]
$$
这就是所谓的行并行方式.
为了计算
$$
Z = [Y_1 ~ Y_2] \left[\begin{matrix} B_1 \\ B_2 \end{matrix} \right]
$$
我们首先在每个处理器上计算 $Y_iB_i$ 然后使用一个all-reduce操作将结果汇总为 $Z=Y_1B_1+Y_2B_2$。

我们还需要注意，在后向计算中，列并行线性层需要聚合输入张量 $X$, 因为在每个处理器 $i$ 上，我们只有 $\dot{X_i}=\dot{Y_i}A_i^T$，因此，我们在各处理器之间进行all-reduce，得到 $\dot{X}=\dot{Y}A^T=\dot{Y_1}A_1^T+\dot{Y_2}A_2^T$。

## 效率
给定 $P$ 个处理器, 我们展现理论上的计算和内存成本，以及基于环形算法的1D张量并行的前向和后向的通信成本。

| 计算 | 内存 (参数) | 内存 (activations) | 通信 (带宽) | 通信 (时延) |
| :-:         | :-:              | :-:                  | :-:                       | :-:                     |
| $O(1/P)$    | $O(1/P)$         | $O(1)$               | $O(2(P-1)/P)$             | $O(2(P-1))$             |


## 使用

在ColossalAI最新的版本中，1D张量并行由`Shardformer`功能实现。
关于`Shardformer`的原理和用法细节请参考当前目录下的Shardformer文档。

<!-- doc-test-command: echo  -->

# 1D Tensor Parallelism

Author: Zhengda Bian, Yongbin Li

**Example Code**
- [Tensor Parallelism with Shardformer](https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/shardformer/examples)

**Related Paper**
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://deepakn94.github.io/assets/papers/megatron-sc21.pdf)

## Introduction

Tensor parallelism partitions model weights across multiple devices in order to reduce memory load.
An efficient 1D tensor parallelism implementation was introduced by [Megatron-LM](https://deepakn94.github.io/assets/papers/megatron-sc21.pdf).

Let's take a linear layer as an example, which consists of a GEMM $Y = XA$. Given 2 processors, we split the columns of $A$ into $[A_1 ~ A_2]$, and calculate $Y_i = XA_i$ on each processor, which then forms $[Y_1 ~ Y_2] = [XA_1 ~ XA_2]$. This is called a column-parallel fashion.

When a second linear layer $Z=YB$ follows the column-parallel one, we split $B$ into
$$
\left[\begin{matrix} B_1 \\ B_2 \end{matrix} \right]
$$
which is called a row-parallel fashion.
To calculate
$$
Z = [Y_1 ~ Y_2] \left[\begin{matrix} B_1 \\ B_2 \end{matrix} \right]
$$
we first calculate $Y_iB_i$ on each processor, then use an all-reduce to aggregate the results as $Z=Y_1B_1+Y_2B_2$.

We also need to note that in the backward pass, the column-parallel linear layer needs to aggregate the gradients of the input tensor $X$, because on each processor $i$ we only have $\dot{X_i}=\dot{Y_i}A_i^T$.
Thus, we apply an all-reduce across the processors to get $\dot{X}=\dot{Y}A^T=\dot{Y_1}A_1^T+\dot{Y_2}A_2^T$.

## Efficiency
Given $P$ processors, we present the theoretical computation and memory cost, as well as the communication cost based on the ring algorithm in both the forward and backward pass of 1D tensor parallelism.

| Computation | Memory (parameters) | Memory (activations) | Communication (bandwidth) | Communication (latency) |
| :-:         | :-:              | :-:                  | :-:                       | :-:                     |
| $O(1/P)$    | $O(1/P)$         | $O(1)$               | $O(2(P-1)/P)$             | $O(2(P-1))$             |

## Usage

1D tensor parallelism is implemented by `Shardformer` feature in the newest version of ColossalAI.
For more details about ideas and usages of `Shardformer`, please refer to [Shardformer Doc](./shardformer.md).

<!-- doc-test-command: echo  -->

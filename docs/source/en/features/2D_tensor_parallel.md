# 2D Tensor Parallelism

Author: Zhengda Bian, Yongbin Li

**Prerequisite**
- [1D Tensor Parallelism](./1D_tensor_parallel.md)

**Example Code**
- [ColossalAI-Examples - 2D Tensor Parallelism](https://github.com/hpcaitech/ColossalAI-Examples/blob/main/features/tensor_parallel/README.md)

**Related Paper**
- [An Efficient 2D Method for Training Super-Large Deep Learning Models](https://arxiv.org/pdf/2104.05343.pdf)

## Introduction

1D tensor parallelism does not partition activations, which can also consume a great amount of memory in terms of large-scale models.
To evenly distribute the computation and memory load, [an efficient 2D tensor parallelism algorithm](https://arxiv.org/pdf/2104.05343.pdf) was introduced based on SUMMA (Scalable Universal Matrix Multiplication Algorithm).

Let's still take a linear layer $Y = XA$ as an example.
Given $P=q\times q$ processors (necessary condition), e.g. $q=2$, we split both the input $X$ and weight $A$ into

$$
\left[\begin{matrix} X_{00} & X_{01} \\ X_{10} & X_{11} \end{matrix} \right]
\text{~and~}
\left[\begin{matrix} A_{00} & A_{01} \\ A_{10} & A_{11} \end{matrix} \right].
$$

The calculation includes $q$ steps. When $t=1$, $X_{i0}$ is broadcasted in its row, and $A_{0j}$ is broadcasted in its column. So, we have

$$
\left[\begin{matrix} X_{00},A_{00} & X_{00},A_{01} \\ X_{10},A_{00} & X_{10},A_{01} \end{matrix} \right].
$$

Then we multiply $X_{i0}$ and $A_{0j}$ on each processor $(i, j)$ as

$$
\left[\begin{matrix} X_{00}A_{00} & X_{00}A_{01} \\ X_{10}A_{00} & X_{10}A_{01} \end{matrix} \right] (1).
$$

Similarly, when $t=2$, $X_{i1}$ is broadcasted in its row, $A_{1j}$ is broadcasted in its column, and we multiply them as

$$
\left[\begin{matrix} X_{01}A_{10} & X_{01}A_{11} \\ X_{11}A_{10} & X_{11}A_{11} \end{matrix} \right] (2).
$$

By adding $(1)$ and $(2)$ up, we have

$$
Y = XA = \left[\begin{matrix} X_{00}A_{00}+X_{01}A_{10} & X_{00}A_{01}+X_{01}A_{11} \\ X_{10}A_{00}+X_{11}A_{10} & X_{10}A_{01}+X_{11}A_{11} \end{matrix} \right].
$$

## Efficiency
Given $P=q\times q$ processors, we present the theoretical computation and memory cost, as well as the communication cost based on the ring algorithm in both the forward and backward pass of 2D tensor parallelism.

| Computation | Memory (parameters) | Memory (activations) | Communication (bandwidth) | Communication (latency) |
| :-:         | :-:              | :-:                  | :-:                       | :-:                     |
| $O(1/q^2)$  | $O(1/q^2)$       | $O(1/q^2)$           | $O(6(q-1)/q)$             | $O(6(q-1))$             |

## Usage

Currently the newest version of ColossalAI doesn't support 2D tensor parallelism, but this feature will be integrated into `Shardformer` in future releases.
For more details about ideas and usages of `Shardformer`, please refer to [Shardformer Doc](./shardformer.md).

For users of older version of ColossalAI, please refer to [ColossalAI-Examples - 2D Tensor Parallelism](https://github.com/hpcaitech/ColossalAI-Examples/blob/main/features/tensor_parallel/README.md).

<!-- doc-test-command: echo  -->

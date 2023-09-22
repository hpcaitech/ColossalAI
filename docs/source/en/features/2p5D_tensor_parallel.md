# 2.5D Tensor Parallelism

Author: Zhengda Bian, Yongbin Li

**Prerequisite**
- [1D Tensor Parallelism](./1D_tensor_parallel.md)
- [2D Tensor Parallelism](./2D_tensor_parallel.md)

**Example Code**
- [ColossalAI-Examples - 2.5D Tensor Parallelism](https://github.com/hpcaitech/ColossalAI-Examples/blob/main/features/tensor_parallel/README.md)

**Related Paper**
- [2.5-dimensional distributed model training](https://arxiv.org/pdf/2105.14500.pdf)

## Introduction

Compared with 1D tensor parallelism, 2D parallelism reduces the memory cost, but may introduce more communication.
Therefore, a  [2.5D tensor parallelism algorithm](https://arxiv.org/pdf/2105.14500.pdf) was proposed based on 2.5D SUMMA to reduce communication by using more devices.

Let's still take a linear layer $Y = XA$ as an example.
Given $P=q \times q \times d$ processors (necessary condition), e.g. $q=d=2$, we split the input $X$ into $d\times q$ rows and $q$ columns as

$$
\left[\begin{matrix} X_{00} & X_{01} \\ X_{10} & X_{11} \\ X_{20} & X_{21} \\ X_{30} & X_{31}\end{matrix} \right],
$$

which can be reshaped into $d$ layers as

$$
\left[\begin{matrix} X_{00} & X_{01} \\ X_{10} & X_{11} \end{matrix} \right] \text{~and~}\left[\begin{matrix} X_{20} & X_{21} \\ X_{30} & X_{31} \end{matrix} \right].
$$

Also, the weight $A$ is split into

$$
\left[\begin{matrix} A_{00} & A_{01} \\ A_{10} & A_{11} \end{matrix} \right].
$$

For each layer of $X$, we use the SUMMA algorithm to multiply $X$ and $A$.
Then, we have the output

$$
\left[\begin{matrix} Y_{00}=X_{00}A_{00}+X_{01}A_{10} & Y_{01}=X_{00}A_{01}+X_{01}A_{11} \\ Y_{10}=X_{10}A_{00}+X_{11}A_{10} & Y_{11}=X_{10}A_{01}+X_{11}A_{11} \end{matrix} \right]
\text{~and~}
$$
$$
\left[\begin{matrix} Y_{20}=X_{20}A_{00}+X_{21}A_{10} & Y_{21}=X_{20}A_{01}+X_{21}A_{11} \\ Y_{30}=X_{30}A_{00}+X_{31}A_{10} & Y_{31}=X_{30}A_{01}+X_{31}A_{11} \end{matrix} \right].
$$

## Efficiency
Given $P=q \times q \times d$ processors, we present the theoretical computation and memory cost, as well as the communication cost based on the ring algorithm in both the forward and backward pass of 2.5D tensor parallelism.

| Computation | Memory (parameters) | Memory (activations) | Communication (bandwidth) | Communication (latency) |
| :-:         | :-:              | :-:                  | :-:                       | :-:                     |
| $O(1/dq^2)$ | $O(1/q^2)$       | $O(1/dq^2)$          | $\small O(3(q-1)(d+1)/dq)$       | $O(6(q-1))$             |

## Usage

Currently the newest version of ColossalAI doesn't support 2.5D tensor parallelism, but this feature will be integrated into `Shardformer` in future releases.
For more details about ideas and usages of `Shardformer`, please refer to [Shardformer Doc](./shardformer.md).

For users of older version of ColossalAI, please refer to [ColossalAI-Examples - 2.5D Tensor Parallelism](https://github.com/hpcaitech/ColossalAI-Examples/blob/main/features/tensor_parallel/README.md).

<!-- doc-test-command: echo  -->

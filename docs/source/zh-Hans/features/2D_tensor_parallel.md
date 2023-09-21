# 2D 张量并行

作者: Zhengda Bian, Yongbin Li

**前置教程**
- [1D 张量并行](./1D_tensor_parallel.md)

**示例代码**
- [ColossalAI-Examples - 2D Tensor Parallelism](https://github.com/hpcaitech/ColossalAI-Examples/blob/main/features/tensor_parallel/README.md)

**相关论文**
- [An Efficient 2D Method for Training Super-Large Deep Learning Models](https://arxiv.org/pdf/2104.05343.pdf)

## 引言

1D张量并行没有对 activations 进行划分，就大规模模型而言，这也会消耗大量的内存。
为了平均分配计算和内存负荷，在 SUMMA（可扩展的通用矩阵乘法算法）的基础上， [2D张量并行](https://arxiv.org/pdf/2104.05343.pdf) 被引入。

我们还是以线性层 $Y = XA$ 为例。
给定 $P=q\times q$ 个处理器（必要条件）, 如 $q=2$, 我们把输入 $X$ 和权重A $A$ 都划分为

$$
\left[\begin{matrix} X_{00} & X_{01} \\ X_{10} & X_{11} \end{matrix} \right]
\text{~and~}
\left[\begin{matrix} A_{00} & A_{01} \\ A_{10} & A_{11} \end{matrix} \right].
$$

该计算包括 $q$ 步。 当 $t=1$ 时, $X_{i0}$ 在其行中被广播, 而 $A_{0j}$ 在其列中被广播。因此，我们有

$$
\left[\begin{matrix} X_{00},A_{00} & X_{00},A_{01} \\ X_{10},A_{00} & X_{10},A_{01} \end{matrix} \right].
$$

然后我们在每个处理器 $(i, j)$ 上将 $X_{i0}$ 和 $A_{0j}$ 相乘为

$$
\left[\begin{matrix} X_{00}A_{00} & X_{00}A_{01} \\ X_{10}A_{00} & X_{10}A_{01} \end{matrix} \right] (1).
$$

同样，当 $t=2$ 时, $X_{i1}$ 在其行中被广播, $A_{1j}$ 在其列中被广播, 我们将它们相乘为

$$
\left[\begin{matrix} X_{01}A_{10} & X_{01}A_{11} \\ X_{11}A_{10} & X_{11}A_{11} \end{matrix} \right] (2).
$$

通过将 $(1)$ 和 $(2)$ 相加，我们有

$$
Y = XA = \left[\begin{matrix} X_{00}A_{00}+X_{01}A_{10} & X_{00}A_{01}+X_{01}A_{11} \\ X_{10}A_{00}+X_{11}A_{10} & X_{10}A_{01}+X_{11}A_{11} \end{matrix} \right].
$$

## 效率
给定 $P=q\times q$ 个处理器, 我们展现理论上的计算和内存成本，以及基于环形算法的2D张量并行的前向和后向的通信成本。

| 计算 | 内存 (参数) | 内存 (activations) | 通信 (带宽) | 通信 (时延) |
| :-:         | :-:              | :-:                  | :-:                       | :-:                     |
| $O(1/q^2)$  | $O(1/q^2)$       | $O(1/q^2)$           | $O(6(q-1)/q)$             | $O(6(q-1))$             |

## 使用

ColossalAI的最新版本还暂不支持2D张量并行，但2D张量并行的功能会在未来的版本被集成入`Shardformer`中。关于`Shardformer`的原理和用法细节请参考当前目录下的Shardformer文档。

对于老版本ColossalAI的用户，2D张量并行的用法请参考[ColossalAI-Examples - 2D Tensor Parallelism](https://github.com/hpcaitech/ColossalAI-Examples/blob/main/features/tensor_parallel/README.md)。

<!-- doc-test-command: echo  -->

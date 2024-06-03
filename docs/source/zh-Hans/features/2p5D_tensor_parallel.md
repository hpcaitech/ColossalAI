# 2.5D 张量并行

作者: Zhengda Bian, Yongbin Li

**前置教程**
- [1D 张量并行](./1D_tensor_parallel.md)
- [2D 张量并行](./2D_tensor_parallel.md)

**示例代码**
- [ColossalAI-Examples - 2.5D Tensor Parallelism](https://github.com/hpcaitech/ColossalAI-Examples/blob/main/features/tensor_parallel/README.md)

**相关论文**
- [2.5-dimensional distributed model training](https://arxiv.org/pdf/2105.14500.pdf)

## 引言

与一维张量并行相比，二维并行降低了内存成本，但可能引入更多的通信。因此，[2.5D张量并行](https://arxiv.org/pdf/2105.14500.pdf) 在 2.5D SUMMA 的基础上被提出，它通过使用更多的设备来减少通信。

我们还是以线性层 $Y = XA$ 为例。
给定 $P=q \times q \times d$ 个处理器（必要条件）, 如 $q=d=2$, 我们把输入 $X$ 划分为 $d\times q$ 行和 $q$ 列

$$
\left[\begin{matrix} X_{00} & X_{01} \\ X_{10} & X_{11} \\ X_{20} & X_{21} \\ X_{30} & X_{31}\end{matrix} \right],
$$
它可以被重塑为 $d$ 层

$$
\left[\begin{matrix} X_{00} & X_{01} \\ X_{10} & X_{11} \end{matrix} \right] \text{~and~}\left[\begin{matrix} X_{20} & X_{21} \\ X_{30} & X_{31} \end{matrix} \right].
$$

另外，权重 $A$ 被分割为

$$
\left[\begin{matrix} A_{00} & A_{01} \\ A_{10} & A_{11} \end{matrix} \right].
$$

对于 $X$ 相关的每一层, 我们使用SUMMA算法将 $X$ 与 $A$ 相乘。
然后，我们得到输出

$$
\left[\begin{matrix} Y_{00}=X_{00}A_{00}+X_{01}A_{10} & Y_{01}=X_{00}A_{01}+X_{01}A_{11} \\ Y_{10}=X_{10}A_{00}+X_{11}A_{10} & Y_{11}=X_{10}A_{01}+X_{11}A_{11} \end{matrix} \right]
\text{~and~}
$$
$$
\left[\begin{matrix} Y_{20}=X_{20}A_{00}+X_{21}A_{10} & Y_{21}=X_{20}A_{01}+X_{21}A_{11} \\ Y_{30}=X_{30}A_{00}+X_{31}A_{10} & Y_{31}=X_{30}A_{01}+X_{31}A_{11} \end{matrix} \right].
$$

## 效率

给定 $P=q \times q \times d$ 个处理器, 我们展现理论上的计算和内存成本，以及基于环形算法的2.5D张量并行的前向和后向的通信成本。

| 计算 | 内存 (参数) | 内存 (activations) | 通信 (带宽) | 通信 (时延) |
| :-:         | :-:              | :-:                  | :-:                       | :-:                     |
| $O(1/dq^2)$ | $O(1/q^2)$       | $O(1/dq^2)$          | $\small O(3(q-1)(d+1)/dq)$       | $O(6(q-1))$             |

## 使用

ColossalAI的最新版本还暂不支持2.5D张量并行，但2.5D张量并行的功能会在未来的版本被集成入`Shardformer`中。关于`Shardformer`的原理和用法细节请参考当前目录下的Shardformer文档。

对于老版本ColossalAI的用户，2.5D张量并行的用法请参考[ColossalAI-Examples - 2.5D Tensor Parallelism](https://github.com/hpcaitech/ColossalAI-Examples/blob/main/features/tensor_parallel/README.md)。

<!-- doc-test-command: echo  -->

# 2.5D 张量并行

作者: Zhengda Bian, Yongbin Li

**前置教程**
- [定义配置文件](../basics/define_your_config.md)
- [并行配置](../basics/configure_parallelization.md)
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

为了使我们的模型能够实现2.5D张量并行，例如在8个 GPU 上，我们需要配置如下的并行设置。

```python
CONFIG = dict(parallel=dict(
    data=1,
    pipeline=1,
    tensor=dict(size=8, mode='2.5d', depth=2),
))

```

然后 Colossal-AI 会自动对所有来自 `colossalai.nn` 的层应用2.5D张量并行。

让我们定义一个由两层多层感知器 (MLP) 组成的模型，如下所示。

```python
import colossalai
import colossalai.nn as col_nn
import torch
from colossalai.utils import print_rank_0

class MLP(torch.nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        intermediate_dim = dim * 4
        self.dense_1 = col_nn.Linear(dim, intermediate_dim)
        print_rank_0(f'Weight of the first linear layer: {self.dense_1.weight.shape}')
        self.activation = torch.nn.GELU()
        self.dense_2 = col_nn.Linear(intermediate_dim, dim)
        print_rank_0(f'Weight of the second linear layer: {self.dense_2.weight.shape}')
        self.dropout = col_nn.Dropout(0.1)

    def forward(self, x):
        x = self.dense_1(x)
        print_rank_0(f'Output of the first linear layer: {x.shape}')
        x = self.activation(x)
        x = self.dense_2(x)
        print_rank_0(f'Output of the second linear layer: {x.shape}')
        x = self.dropout(x)
        return x
```
在8个 GPU 上启动 Colossal-AI 并建立模型。
```python
parser = colossalai.get_default_parser()
colossalai.launch(config=CONFIG,
                  rank=args.rank,
                  world_size=args.world_size,
                  local_rank=args.local_rank,
                  host=args.host,
                  port=args.port)

m = MLP()
```
我们将会看到 MLP 模型中被划分的参数（如权重）的形状。
```shell
Weight of the first linear layer: torch.Size([128, 512])
Weight of the second linear layer: torch.Size([512, 128])
```

第一个线性层的完整权重形状应该为 `[256, 1024]`. 经过2.5D并行划分后，它在每个 GPU 上变成了 `[128, 512]` 。
同样地，第二层将权重 `[1024, 256]` 划分为 `[512, 128]`.

我们可以用一些随机输入来运行这个模型。
```python
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import get_current_device

x = torch.randn((16, 256), device=get_current_device())
# partition input
torch.distributed.broadcast(x, src=0)
x = torch.chunk(x, 2, dim=0)[gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP)]
x = torch.chunk(x, 2, dim=0)[gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)]
x = torch.chunk(x, 2, dim=-1)[gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)]
print_rank_0(f'Input: {x.shape}')

x = m(x)
```
然后我们可以看到 activation 结果的形状。
```shell
Input: torch.Size([4, 128])
Output of the first linear layer: torch.Size([4, 512])
Output of the second linear layer: torch.Size([4, 128])
```
2.5D并行中的 activation 张量都是同时在$d \times q$行和$q$列分割的。例如，第一个线性层的输出是 `[4, 512]`, 而第二层的输出为 `[4, 128]`。
注意，2.5D并行使用与2D并行相同的划分方法来处理权重，区别在于对输入的划分。

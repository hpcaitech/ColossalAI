# 1D 张量并行

作者: Zhengda Bian, Yongbin Li

**前置教程**
- [定义配置文件](../basics/define_your_config.md)
- [并行配置](../basics/configure_parallelization.md)

**示例代码**
- [ColossalAI-Examples 1D Tensor Parallelism](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/tensor_parallel/tensor_parallel_1d.py)

**相关论文**
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://deepakn94.github.io/assets/papers/megatron-sc21.pdf)

## 引言

张量并行将模型参数划分到多个设备上，以减少内存负荷。
[Megatron-LM](https://deepakn94.github.io/assets/papers/megatron-sc21.pdf) 介绍了一种高效的一维张量并行化实现。

让我们以一个线性层为例，它包括一个 GEMM $Y = XA$。 给定2个处理器，我们把列 $A$ 划分为 $[A_1 ~ A_2]$, 并在每个处理器上计算 $Y_i = XA_i$ , 然后形成 $[Y_1 ~ Y_2] = [XA_1 ~ XA_2]$. 这被称为列并行方式。

当第二个线性层 $Z=YB$ 跟随上述列并行层的时候, 我们把 $B$ 划分为
```math
\left[\begin{matrix} B_1 \\ B_2 \end{matrix} \right]
```
这就是所谓的行并行方式.<br>

为了计算
```math
Z = [Y_1 ~ Y_2] \left[\begin{matrix} B_1 \\ B_2 \end{matrix} \right]
```
我们首先在每个处理器上计算 $Y_iB_i$ 然后使用一个all-reduce操作将结果汇总为 $Z=Y_1B_1+Y_2B_2$。

我们还需要注意，在后向计算中，列并行线性层需要聚合输入张量 $X$, 因为在每个处理器 $i$ 上，我们只有 $\dot{X_i}=\dot{Y_i}A_i^T$，因此，我们在各处理器之间进行all-reduce，得到 $\dot{X}=\dot{Y}A^T=\dot{Y_1}A_1^T+\dot{Y_2}A_2^T$。

## 效率
给定 $P$ 个处理器, 我们展现理论上的计算和内存成本，以及基于环形算法的1D张量并行的前向和后向的通信成本。

| 计算 | 内存 (参数) | 内存 (activations) | 通信 (带宽) | 通信 (时延) |
| :-:         | :-:              | :-:                  | :-:                       | :-:                     |
| $O(1/P)$    | $O(1/P)$         | $O(1)$               | $O(2(P-1)/P)$             | $O(2(P-1))$             |

## 使用

为了使模型能够实现一维张量并行, 如在2个 GPU 上, 我们需要配置如下的并行设置。
```python
CONFIG = dict(parallel=dict(
    data=1,
    pipeline=1,
    tensor=dict(size=2, mode='1d'),
))
```

然后 Colossal-AI 会自动对所有来自 `colossalai.nn` 的层应用1D张量并行。

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
        print_rank_0(f'Weight of the first linear layer: {self.dense_1.weight.transpose(0, 1).shape}')
        self.activation = torch.nn.GELU()
        self.dense_2 = col_nn.Linear(intermediate_dim, dim)
        print_rank_0(f'Weight of the second linear layer: {self.dense_2.weight.transpose(0, 1).shape}')
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

在2个 GPU 上启动 Colossal-AI 并建立模型。

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
Weight of the first linear layer: torch.Size([256, 512])
Weight of the second linear layer: torch.Size([512, 256])
```
第一个线性层的完整权重形状应该为 `[256, 1024]`. 经过列-并行分割，它变成了 `[256, 512]`。
同样地，第二个行并行层将权重 `[1024, 256]` 划分为 `[512, 256]`。

我们可以用一些随机输入来运行这个模型。
```python
from colossalai.utils import get_current_device

x = torch.randn((16, 256), device=get_current_device())
torch.distributed.broadcast(x, src=0)  # synchronize input

x = m(x)
```
然后我们可以看到 activation 结果的形状。
```shell
Output of the first linear layer: torch.Size([16, 512])
Output of the second linear layer: torch.Size([16, 256])
```
第一个线性层的输出被划分成2块 (每个形状为 `[16, 512]`), 而第二层在整个 GPU 上的输出是相同的。

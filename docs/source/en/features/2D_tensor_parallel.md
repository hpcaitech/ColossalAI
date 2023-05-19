# 2D Tensor Parallelism

Author: Zhengda Bian, Yongbin Li

**Prerequisite**
- [Define Your Configuration](../basics/define_your_config.md)
- [Configure Parallelization](../basics/configure_parallelization.md)
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

To enable 2D tensor parallelism for our model, e.g. on 4 GPUs, we need to configure the parallelism setting as below.
```python
CONFIG = dict(parallel=dict(
    data=1,
    pipeline=1,
    tensor=dict(size=4, mode='2d'),
))
```
Then Colossal-AI will automatically apply 2D parallelism to all the layers from `colossalai.nn`.

Let's define a model that consists of a two-layer multi-layer perceptron (MLP) as below.
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
Launch Colossal-AI on 4 GPUs and build the model
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
We will see the shapes of partitioned parameters(e.g. weights) in the MLP model.
```shell
Weight of the first linear layer: torch.Size([128, 512])
Weight of the second linear layer: torch.Size([512, 128])
```
The complete weight of the first linear layer is supposed to have the shape `[256, 1024]`. After the partitioning of 2D parallelism, it becomes `[128, 512]` on each GPU.
Similarly, the second layer partitions the weight `[1024, 256]` into `[512, 128]`.

We can run the model with some random inputs.
```python
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import get_current_device

x = torch.randn((16, 256), device=get_current_device())
# partition input
torch.distributed.broadcast(x, src=0)
x = torch.chunk(x, 2, dim=0)[gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL)]
x = torch.chunk(x, 2, dim=-1)[gpc.get_local_rank(ParallelMode.PARALLEL_2D_ROW)]
print_rank_0(f'Input: {x.shape}')

x = m(x)
```
Then we can see the shapes of activation results.
```shell
Input: torch.Size([8, 128])
Output of the first linear layer: torch.Size([8, 512])
Output of the second linear layer: torch.Size([8, 128])
```
The activation tensors in 2D parallelism are all split in both row and column.
E.g. the output of the first linear layer has the shape `[8, 512]`, while the second layer has the output of `[8, 128]`.

# 2.5D Tensor Parallelism

Author: Zhengda Bian, Yongbin Li

**Prerequisite**
- [Define Your Configuration](../basics/define_your_config.md)
- [Configure Parallelization](../basics/configure_parallelization.md)
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

To enable 2.5D tensor parallelism for our model, e.g. on 8 GPUs, we need to configure the parallelism setting as below.
```python
CONFIG = dict(parallel=dict(
    data=1,
    pipeline=1,
    tensor=dict(size=8, mode='2.5d', depth=2),
))

```
Then Colossal-AI will automatically apply 2.5D parallelism to all the layers from `colossalai.nn`.

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
Launch Colossal-AI on 8 GPUs and build the model
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
The complete weight of the first linear layer is supposed to have the shape `[256, 1024]`. After the partitioning of 2.5D parallelism, it becomes `[128, 512]` on each GPU.
Similarly, the second layer partitions the weight `[1024, 256]` into `[512, 128]`.

We can run the model with some random inputs.
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
Then we can see the shapes of activation results.
```shell
Input: torch.Size([4, 128])
Output of the first linear layer: torch.Size([4, 512])
Output of the second linear layer: torch.Size([4, 128])
```
The activation tensors in 2.5D parallelism are all split by $d \times q$ in the row and $q$ in the column.
E.g. the output of the first linear layer has the shape `[4, 512]`), while the second layer has the output of `[4, 128]`.
Note, 2.5D parallelism use the same partition method as 2D parallelism for weights, where the difference is the partition of input.

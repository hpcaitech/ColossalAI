# 3D Tensor Parallelism

Author: Zhengda Bian, Yongbin Li

**Prerequisite**
- [Define Your Configuration](../basics/define_your_config.md)
- [Configure Parallelization](../basics/configure_parallelization.md)
- [1D Tensor Parallelism](./1D_tensor_parallel.md)
- [2D Tensor Parallelism](./2D_tensor_parallel.md)

**Example Code**
- [ColossalAI-Examples - 3D Tensor Parallelism](https://github.com/hpcaitech/ColossalAI-Examples/blob/main/features/tensor_parallel/README.md)

**Related Paper**
- [Maximizing Parallelism in Distributed Training for Huge Neural Networks](https://arxiv.org/pdf/2105.14450.pdf)

## Introduction

The [3D tensor parallelism](https://arxiv.org/pdf/2105.14450.pdf) is an approach to parallelize the computation of neural models, hoping to obtain the optimal communication cost.

Let's still take a linear layer $Y = XA$ as an example.
Given $P=q \times q \times q$ processors (necessary condition), e.g. $q=2$, we split the input $X$ and weight $A$ into

$$
\left[\begin{matrix}
            X_{000} & X_{001} \\
            X_{010} & X_{011} \\
            X_{100} & X_{101} \\
            X_{110} & X_{111} \end{matrix}
\right]
\text{~and~}
\left[\begin{matrix}
            A_{000} & A_{001} & A_{010} & A_{011} \\
            A_{100} & A_{101} & A_{110} & A_{111} \end{matrix}
\right]
\text{~respectively,}$$
where each $X_{ijl}$ and $A_{lji}$ are stored at processor $(i,j,l)$, as shown in the figure below.

<center>
<img src="https://s2.loli.net/2022/02/17/JevO6SED5z4PFdp.png" width = "200" height = "250" />
<img src="https://s2.loli.net/2022/02/17/qvtwjdfNXMAb4nF.png" width = "200" height = "250" />
<img src="https://s2.loli.net/2022/02/17/WFzm2N4IwKf1jXZ.png" width = "200" height = "250" />
<img src="https://s2.loli.net/2022/02/17/r2dZQ4hKxwTuIv6.png" width = "200" height = "250" />
</center>

Then we all-gather $X_{ijl}$ across $(i, 0...q,l)$, as well as $A_{lji}$ across $(0...q, j, l)$.
So, we have $X_{il}$ and $A_{lj}$ on each processor $(i,j,l)$ to get $X_{il}A_{lj}$.
Finally, we reduce-scatter the results across $(i, j, 0...q)$ to get $Y_{ijl}$, which forms
$$
Y=
\left[\begin{matrix}
            Y_{000} & Y_{001} \\
            Y_{010} & Y_{011} \\
            Y_{100} & Y_{101} \\
            Y_{110} & Y_{111} \end{matrix}
\right].
$$

We also need to note that in the backward pass, we need to all-gather the gradient $\dot{Y_{ijl}}$, and then reduce-scatter the gradient $\dot{X_{il}}=\dot{Y_{ij}}A_{lj}^T$ and $\dot{A_{lj}}=X_{il}^T\dot{Y_{ij}}$.

## Efficiency
Given $P=q \times q \times q$ processors, we present the theoretical computation and memory cost, as well as the communication cost based on the ring algorithm in both the forward and backward pass of 3D tensor parallelism.

| Computation | Memory (parameters) | Memory (activations) | Communication (bandwidth) | Communication (latency) |
| :-:         | :-:              | :-:                  | :-:                       | :-:                     |
| $O(1/q^3)$  | $O(1/q^3)$       | $O(1/q^3)$           | $O(6(q-1)/q^3)$           | $O(6(q-1))$             |

## Usage

To enable 3D tensor parallelism for our model, e.g. on 8 GPUs, we need to configure the parallelism setting as below.
```python
CONFIG = dict(parallel=dict(
    data=1,
    pipeline=1,
    tensor=dict(size=8, mode='3d'),
))
```
Then Colossal-AI will automatically apply 3D parallelism to all the layers from `colossalai.nn`.

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
Weight of the first linear layer: torch.Size([128, 256])
Weight of the second linear layer: torch.Size([512, 64])
```
The complete weight of the first linear layer is supposed to have the shape `[256, 1024]`. After the partitioning of 3D parallelism, it becomes `[128, 256]` on each GPU.
Similarly, the second layer partitions the weight `[1024, 256]` into `[512, 64]`.

We can run the model with some random inputs.
```python
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import get_current_device

x = torch.randn((16, 256), device=get_current_device())
# partition input
torch.distributed.broadcast(x, src=0)
x = torch.chunk(x, 2, dim=0)[gpc.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)]
x = torch.chunk(x, 2, dim=0)[gpc.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)]
x = torch.chunk(x, 2, dim=-1)[gpc.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)]
print_rank_0(f'Input: {x.shape}')

x = m(x)
```
Then we can see the shapes of activation results.
```shell
Input: torch.Size([4, 128])
Output of the first linear layer: torch.Size([4, 512])
Output of the second linear layer: torch.Size([4, 128])
```
The activation tensors in 3D parallelism are all split by $q^2$ in the row and $q$ in the column.
E.g. the output of the first linear layer has the shape `[4, 512]`), while the second layer has the output of `[4, 128]`.
Note, although the results of 3D parallelism have the same shape as that of 2.5D parallelism for weights here, the content of each partition is different.

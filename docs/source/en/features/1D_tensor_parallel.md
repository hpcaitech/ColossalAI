# 1D Tensor Parallelism

Author: Zhengda Bian, Yongbin Li

**Prerequisite**
- [Define Your Configuration](../basics/define_your_config.md)
- [Configure Parallelization](../basics/configure_parallelization.md)

**Example Code**
- [ColossalAI-Examples 1D Tensor Parallelism](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/tensor_parallel/tensor_parallel_1d.py)

**Related Paper**
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://deepakn94.github.io/assets/papers/megatron-sc21.pdf)

## Introduction

Tensor parallelism partitions model weights across multiple devices in order to reduce memory load.
An efficient 1D tensor parallelism implementation was introduced by [Megatron-LM](https://deepakn94.github.io/assets/papers/megatron-sc21.pdf).

Let's take a linear layer as an example, which consists of a GEMM $Y = XA$. Given 2 processors, we split the columns of $A$ into $[A_1 ~ A_2]$, and calculate $Y_i = XA_i$ on each processor, which then forms $[Y_1 ~ Y_2] = [XA_1 ~ XA_2]$. This is called a column-parallel fashion.

When a second linear layer $Z=YB$ follows the column-parallel one, we split $B$ into 
```math
\left[\begin{matrix} B_1 \\ B_2 \end{matrix} \right]
```
which is called a row-parallel fashion.
To calculate 
```math
Z = [Y_1 ~ Y_2] \left[\begin{matrix} B_1 \\ B_2 \end{matrix} \right]
```
we first calculate $Y_iB_i$ on each processor, then use an all-reduce to aggregate the results as $Z=Y_1B_1+Y_2B_2$.

We also need to note that in the backward pass, the column-parallel linear layer needs to aggregate the gradients of the input tensor $X$, because on each processor $i$ we only have $\dot{X_i}=\dot{Y_i}A_i^T$.
Thus, we apply an all-reduce across the processors to get $\dot{X}=\dot{Y}A^T=\dot{Y_1}A_1^T+\dot{Y_2}A_2^T$.

## Efficiency
Given $P$ processors, we present the theoretical computation and memory cost, as well as the communication cost based on the ring algorithm in both the forward and backward pass of 1D tensor parallelism.

| Computation | Memory (parameters) | Memory (activations) | Communication (bandwidth) | Communication (latency) |
| :-:         | :-:              | :-:                  | :-:                       | :-:                     |
| $O(1/P)$    | $O(1/P)$         | $O(1)$               | $O(2(P-1)/P)$             | $O(2(P-1))$             |

## Usage

To enable 1D tensor parallelism for our model, e.g. on 2 GPUs, we need to configure the parallism setting as below.
```python
CONFIG = dict(parallel=dict(
    data=1,
    pipeline=1,
    tensor=dict(size=2, mode='1d'),
))
```
Then Colossal-AI will automatically apply 1D parallelism to all the layers from `colossalai.nn`.

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

Launch Colossal-AI on 2 GPUs and build the model.

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
Weight of the first linear layer: torch.Size([256, 512])
Weight of the second linear layer: torch.Size([512, 256])
```
The complete weight of the first linear layer is supposed to have the shape `[256, 1024]`. After the column-parallel partitioning, it becomes `[256, 512]`.
Similarly, the second row-parallel layer partitions the weight `[1024, 256]` into `[512, 256]`.

We can run the model with some random inputs.
```python
from colossalai.utils import get_current_device

x = torch.randn((16, 256), device=get_current_device())
torch.distributed.broadcast(x, src=0)  # synchronize input

x = m(x)
```
Then we can see the shapes of activation results.
```shell
Output of the first linear layer: torch.Size([16, 512])
Output of the second linear layer: torch.Size([16, 256])
```
The output of the first linear layer is split into 2 partitions (each has the shape `[16, 512]`), while the second layer has identical outputs across the GPUs.

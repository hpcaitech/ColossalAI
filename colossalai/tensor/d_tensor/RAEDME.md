# 🔢 Distributed Tensor

## 📚 Table of Contents

- [🔢 Distributed Tensor](#-distributed-tensor)
  - [📚 Table of Contents](#-table-of-contents)
  - [🔗 Introduction](#-introduction)
  - [📝 Design](#-design)
  - [🔨 Usage](#-usage)
  - [🎈 Progress Log](#-progress-log)

## 🔗 Introduction

Distributed tensor is a type of tensor that is distributed across multiple devices. It is a wrapper of PyTorch tensor, and it is used to support distributed training.
It can represent the device topology and tensor placement over the devices in the topology. It also provides a set of APIs to manipulate the distributed tensor.

## 📝 Design

Our implementation is inspired by the work [Alpa](https://arxiv.org/abs/2201.12023), which unifies data parallelism and tensor parallelism as intra-op parallelism. It uses notations `S` to represent the sharded dimension and `R` to represent the replicated dimension. For example, given a 2D matrix, `[S, R]` represents the tensor is sharded over the first dimension.

Each sharded dimension will have a subscript to represent its placement over the devices. Assuming we have 4 GPUs and the GPUs are arranged in a 2 x 2 manner. Let's say we have a 2D matrix like below:


```text
    [1,  2,  3,  4 ]
A = [4,  5,  6,  7 ]
    [8,  9,  10, 11]
    [12, 13, 14, 15]
```

`[S0, R]` would mean that the first dimension is sharded over the rows in the device topology.

```text
| --------------------—————————————————————-|
|                     |                     |
|  [1,  2,  3,  4 ]   |  [1,  2,  3,  4 ]   |
|  [4,  5,  6,  7 ]   |  [4,  5,  6,  7 ]   |
|                     |                     |
| --------------------——————————————————-----
|                     |                     |
|  [8,  9,  10, 11]   |  [8,  9,  10, 11]   |
|  [12, 13, 14, 15]   |  [12, 13, 14, 15]   |
|                     |                     |
| --------------------——————————————————-----
```

`[S01, R]` would mean that the first dimension is sharded over both the row and column in the device topology.

```text
| --------------------—————————————————————-|
|                     |                     |
|  [1,  2,  3,  4 ]   |  [4,  5,  6,  7 ]   |
|                     |                     |
| --------------------——————————————————-----
|                     |                     |
|  [8,  9,  10, 11]   |  [12, 13, 14, 15]   |
|                     |                     |
| --------------------——————————————————-----
```

## 🔨 Usage

A sample API usage is given below.

```python
import torch

import colossalai
from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.d_tensor import DTensor, ShardingSpec

colossalai.launch_from_torch(config={})

# define your device mesh
# assume you have 4 GPUs
physical_mesh_id = torch.arange(0, 4)
mesh_shape = (2, 2)
device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

# define a tensor
a = torch.rand(16, 32).cuda()

# create sharding spec for the tensor
# assume the sharding spec is [S0, R]
dim_partition_dict = {0: [0]}
sharding_spec = ShardingSpec(a.dim(), dim_partition_dict)

# create a distributed tensor
d_tensor = DTensor(a, device_mesh, sharding_spec)
print(d_tensor)

global_tensor = d_tensor.to_global()
print(global_tensor)
```


## 🎈 Progress Log

- [x] Support layout conversion
- [x] Support sharding on 2D device mesh
- [ ] Support sharding on 3D device mesh
- [ ] Support sharding 4D device mesh
- [ ] Support sharding info saving and offline tensor merge (we can save tensor as dtensor and gather the tensors back to the global tensor based on the sharding info in a single process in CPU, useful for distributed training checkpoint load and save.)

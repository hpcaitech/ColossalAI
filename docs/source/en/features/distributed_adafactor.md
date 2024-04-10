# Distributed Adafactor

Author: 

**Related Paper**
- [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)

## Introduction

Distributed Adafactor is an optimiser that supports hybrid optimisation, including 1D tensor parallelism as well as ZerO. It makes full use of computational resources through reasonable task parallelism, improves training efficiency and speed, and reduces space pressure on single card storage. It has a wide range of applications and currently supports a range of Transformer based models, see [tests.kit.model_zoo](https://github.com/hpcaitech/ColossalAI/tree/main/tests/kit/model_zoo) for details. 


## Performance

|        Parallel strategy        |    iter    | Float Percision |      Device Nums     | weight shape  | Avg runtime(ms)  | Avg Speed Up Rate | Best Speed Up Rate  |
| :-----------------------------: | :--------: | :-------------: | :------------------: | :-----------: | :--------------: | :-----------------: | :---------------: |
|           AdaFactor             |     50     |     float32     |          2           | [4096 , 4096] |        0.58      |           -         |          -        |
|    DistAdaFactor(Col Parallel)  |     50     |     float32     |          2           | [4096 , 4096] |        0.41      |         1.39        |        56.91      |
|    DistAdaFactor(Col Parallel)  |     50     |     float32     |          2           | [4096 , 4096] |        0.61      |         0.96        |        18.69      |
|           AdaFactor             |     50     |     float16     |          2           | [4096 , 4096] |        0.72      |           -         |          -        |
|    DistAdaFactor(Col Parallel)  |     50     |     float16     |          2           | [4096 , 4096] |        0.54      |         1.33        |        26.03      |
|    DistAdaFactor(Row Parallel)  |     50     |     float16     |          2           | [4096 , 4096] |        0.67      |         1.08        |        20.55      |
|           AdaFactor             |     50     |     bfloat16    |          2           | [4096 , 4096] |        0.72      |           -         |          -        |
|    DistAdaFactor(Col Parallel)  |     50     |     bfloat16    |          2           | [4096 , 4096] |        0.55      |         1.31        |        26.11      |
|    DistAdaFactor(Row Parallel)  |     50     |     bfloat16    |          2           | [4096 , 4096] |        0.67      |         1.07        |        21.86      |
| :-----------------------------: | :--------: | :-------------: | :------------------: | :-----------: | :--------------: | :-----------------: | :---------------: |
|           AdaFactor             |     50     |     float32     |          4           | [4096 , 4096] |        0.57      |           -         |          -        |
|    DistAdaFactor(Col Parallel)  |     50     |     float32     |          4           | [4096 , 4096] |        0.38      |         1.48        |        53.99      |
|    DistAdaFactor(Col Parallel)  |     50     |     float32     |          4           | [4096 , 4096] |        0.60      |         0.95        |        16.53      |
|           AdaFactor             |     50     |     float16     |          4           | [4096 , 4096] |        0.70      |           -         |          -        |
|    DistAdaFactor(Col Parallel)  |     50     |     float16     |          4           | [4096 , 4096] |        0.50      |         1.44        |        21.98      |
|    DistAdaFactor(Row Parallel)  |     50     |     float16     |          4           | [4096 , 4096] |        0.64      |         1.12        |        15.35      |
|           AdaFactor             |     50     |     bfloat16    |          4           | [4096 , 4096] |        0.72      |           -         |          -        |
|    DistAdaFactor(Col Parallel)  |     50     |     bfloat16    |          4           | [4096 , 4096] |        0.56      |         1.29        |        25.63      |
|    DistAdaFactor(Row Parallel)  |     50     |     bfloat16    |          4           | [4096 , 4096] |        0.71      |         1.09        |        21.52      |
| :-----------------------------: | :--------: | :-------------: | :------------------: | :-----------: | :--------------: | :-----------------: | :---------------: |
|           AdaFactor             |     50     |     float32     |          8           | [4096 , 4096] |        0.56      |           -         |          -        |
|    DistAdaFactor(Col Parallel)  |     50     |     float32     |          8           | [4096 , 4096] |        0.38      |         1.50        |        54.51      |
|    DistAdaFactor(Col Parallel)  |     50     |     float32     |          8           | [4096 , 4096] |        0.91      |         0.67        |        15.68      |
|           AdaFactor             |     50     |     float16     |          8           | [4096 , 4096] |        0.74      |           -         |          -        |
|    DistAdaFactor(Col Parallel)  |     50     |     float16     |          8           | [4096 , 4096] |        0.84      |         0.87        |         9.21      |
|    DistAdaFactor(Row Parallel)  |     50     |     float16     |          8           | [4096 , 4096] |        1.03      |         0.75        |        16.12      |
|           AdaFactor             |     50     |     bfloat16    |          8           | [4096 , 4096] |        0.71      |           -         |          -        |
|    DistAdaFactor(Col Parallel)  |     50     |     bfloat16    |          8           | [4096 , 4096] |        0.54      |         1.31        |        27.28      |
|    DistAdaFactor(Row Parallel)  |     50     |     bfloat16    |          8           | [4096 , 4096] |        0.73      |         1.03        |        25.01      |


## Hands-On Practice
We now demonstrate how to use Distributed Adafactor.  
### step 1. Import libraries

```python
import torch
from torch import nn
import torch.distributed as dist

from colossalai.cluster import ProcessGroupMesh
from colossalai.shardformer.layer import Linear1D_Col, Linear1D_Row
from colossalai.nn.optimizer.distributed_adafactor import DistributedAdaFactor

```

### step 2. Initialize Distributed Environment and Parallism Group
We then need to initialize distributed environment. For demo purpose, we uses `colossalai.launch`. You can refer to [Launch Colossal-AI](../basics/launch_colossalai.md)
for other initialization methods. We use `ProcessGroupMesh` to create tensor parallelism group and data parallelism group.

```python
# Distributed Enviroment
config = {}
colossalai.launch(config=config, rank=rank, world_size=world_size,host="localhost", port=port, backend="nccl")

# Parallism Group
tp_size, zero_size = 2, 2
use_zero = True if zero_size > 1 else False
proc_mesh = ProcessGroupMesh(tp_size, zero_size)
tp_group, dp_group = proc_mesh.get_group_along_axis(0), proc_mesh.get_group_along_axis(1)
```

### step 3. Initialize Module
Build our model. We created an MLP using two Linear Layer.

```python
# Init a Tensor Paralleled Module
class TPModel(nn.Module):
    def __init__(self, linear1, linear2, tp_group=None):
        super().__init__()
        self.linear1 = Linear1D_Col.from_native_module(
            linear1, process_group=tp_group, gather_output=False, overlap=True
        )
        self.linear2 = Linear1D_Row.from_native_module(linear2, process_group=tp_group, parallel_input=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x
HEIGHT = 4096
WIDTH = 4096
tp_model = TPModel(copy.deepcopy(nn.Linear(HEIGHT, WIDTH)), copy.deepcopy(nn.Linear(HEIGHT, WIDTH)), tp_group).to(local_rank)

# Get Module parameter
tp_param_group = [p for n, p in tp_model.named_parameters()]
```

### step 4. Initialize Optimizer
Then, We initialise the optimiser using the model parameter. Then, we set the distributed information for optimiser.

```python
# Init a Optimizer
dist_optim = DistributedAdaFactor(tp_param_group)
shard_to_param = {id(p):p for p in tp_param_group}

# Setup distributed information for Optimizer
dist_optim.setup_distributed(
    tensor_parallel_group=tp_group,
    data_parallel_group=dp_group,
    shard_to_param=shard_to_param,
    use_zero=use_zero,
)
```

### step 5. Perform a forward and backward propagation for model and step the gradient

```python
# Random initialise dataset
x = torch.randn(HEIGHT, WIDTH, device=local_rank)

# Fwd and Bwd
out_tp = tp_model(x)
if zero_size > 1:
    dist_optim.backward(out_tp.sum())
else:
    out_tp.sum().backward()

# perform step for param and grad  
dist_optim.step()
dist_optim.zero_grad()
```
<!-- doc-test-command: echo  -->

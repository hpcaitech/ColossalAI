# Distributed Adafactor

Author: 

**Related Paper**
- [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)

## Introduction

Distributed Adafactor is an optimiser that supports hybrid optimisation, including 1D tensor parallelism as well as ZerO. It makes full use of computational resources through reasonable task parallelism, improves training efficiency and speed, and reduces space pressure on single card storage. It has a wide range of applications and currently supports a range of Transformer based models, see [tests.kit.model_zoo](https://github.com/hpcaitech/ColossalAI/tree/main/tests/kit/model_zoo) for details. 

### API Reference

{{ autodoc:colossalai.nn.optimizer.distributed_adafactor.DistributedAdaFactor }}

## Hands-On Practice
We now demonstrate how to start Distributed Adafactor with booster API.  
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

### step 5.Init Booster

```python
plugin = LowLevelZeroPlugin()
booster = Booster(plugin=plugin)
criterion = lambda x: x.mean()
tp_model, dist_optim, criterion, _, _ = booster.boost(tp_model, dist_optim, criterion) 
```
### step 6.Perform a forward and backward propagation for model and step the gradient

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

## Supporting Information
Model/Feature Compatibility Matrix:
<table>
  <tr>
    <th nowrap="nowrap">Model/Feature</th>
    <th nowrap="nowrap" title="Transformers Bert">Transformers<br />Bert</th>
    <th nowrap="nowrap" align="center" title="Transformers Bert For Pretraining">Transformers Bert<br />For Pretraining</th>
    <th nowrap="nowrap" align="center" title="Transformers Bert Lm Head Model">Transformers Bert<br />Lm Head Model</th>
    <th nowrap="nowrap" align="center" title="Transformers Bert For Masked Lm">Transformers Bert<br />For Masked Lm</th>
    <th nowrap="nowrap" align="center" title="Transformers Bert For Sequence Classification">Transformers Bert<br />For Sequence Classification</th>
    <th nowrap="nowrap" align="center" title="Transformers Bert For Token Classification">Transformers Bert<br />For Token Classification</th>
    <th nowrap="nowrap" align="center" title="Transformers Bert For Next Sentence">Transformers Bert<br />For Next Sentence</th>
    <th nowrap="nowrap" align="center" title="Transformers Bert For Multiple-choice Question">Transformers Bert<br />For Multiple-choice Question</th>
    <th nowrap="nowrap" align="center" title="Transformers Bert For Question Answering">Transformers Bert<br />For Question Answering</th>
  </tr>
  <tr>
    <td nowrap="nowrap">Distributedt<br />Adafactor</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
  </tr>
  
  <tr>
    <td colspan="39"></td>
  </tr>
</table>
<!-- doc-test-command: echo  -->

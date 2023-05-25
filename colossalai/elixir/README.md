# Elixir (Gemini2.0)
Elixir, also known as Gemini, is a technology designed to facilitate the training of large models on a small GPU cluster.
Its goal is to eliminate data redundancy and leverage CPU memory to accommodate really large models.
In addition, Elixir automatically profiles each training step prior to execution and selects the optimal configuration for the ratio of redundancy and the device for each parameter.
This repository is used to benchmark the performance of Elixir.
Elixir will be integrated into ColossalAI for usability.

## Environment

This version is a beta release, so the running environment is somewhat restrictive.
We are only demonstrating our running environment here, as we have not yet tested its compatibility.
We have set the CUDA version to `11.6` and the PyTorch version to `1.13.1+cu11.6`.

## Examples

Here is a simple example to wrap your model and optimizer for [fine-tuning](https://github.com/hpcaitech/Elixir/tree/main/example/fine-tune).

```python
from elixir.search import minimum_waste_search
from elixir.wrapper import ElixirModule, ElixirOptimizer

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-8)

sr = minimum_waste_search(model, world_size)
model = ElixirModule(model, sr, world_group)
optimizer = ElixirOptimizer(model, optimizer)
```

Here is an advanced example for performance, which is used in our [benchmarkhere](https://github.com/hpcaitech/Elixir/blob/main/example/common/elx.py).

```python
import torch
import torch.distributed as dist
from colossalai.nn.optimizer import HybridAdam
from elixir.wrapper import ElixirModule, ElixirOptimizer

# get the world communication group
global_group = dist.GroupMember.WORLD
# get the communication world size
global_size = dist.get_world_size()

# initialize the model in CPU
model = get_model(model_name)
# HybridAdam allows a part of parameters updated on CPU and a part updated on GPU
optimizer = HybridAdam(model.parameters(), lr=1e-3)

sr = optimal_search(
    model,
    global_size,
    unified_dtype=torch.float16,  # enable for FP16 training
    overlap=True,  # enable for overlapping communications
    verbose=True,  # print detailed processing information
    inp=data,  # proivde an example input data in dictionary format
    step_fn=train_step  # provide an example step function
)
model = ElixirModule(
    model,
    sr,
    global_group,
    prefetch=True,  # prefetch chunks to overlap communications
    dtype=torch.float16,  # use AMP
    use_fused_kernels=True  # enable fused kernels in Apex
)
optimizer = ElixirOptimizer(
    model,
    optimizer,
    initial_scale=64,  # loss scale used in AMP
    init_step=True  # enable for the stability of training
)
```

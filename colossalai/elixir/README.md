# ⚡️ Elixir (Gemini2.0)

## 📚 Table of Contents

- [⚡️ Elixir (Gemini2.0)](#️-elixir-gemini20)
  - [📚 Table of Contents](#-table-of-contents)
  - [🔗 Introduction](#-introduction)
  - [💡 Design and Implementation](#-design-and-implementation)
  - [🔨 API Usage](#-api-usage)
    - [General Usage](#general-usage)
    - [Advanced Usage](#advanced-usage)

## 🔗 Introduction

Elixir, also known as Gemini 2.0, is a distributed training technique designed to facilitate large-scale model training on a small GPU cluster.
Its goal is to eliminate data redundancy and leverage CPU memory to accommodate really large models.
Elixir automatically profiles each training step before execution and selects the optimal configuration for the ratio of memory redundancy (tensor sharding) and the device placement for each parameter (tensor offloading).

Please note the following before you try this feature:

- **This feature is still in its experimental stage and the API is subject to future changes.**
- **We have only tested this feature with PyTorch 1.13**


## 💡 Design and Implementation

Existing methods such as DeepSpeed and FSDP often lead to suboptimal efficiency due to the large combination of hyperparameters to tune and only experienced experts can unleash the full potential of hardware by carefully tuning the distributed configuration.
Thus, we present a novel solution, Elixir, which automates efficient large model training based on pre-runtime model profiling.
Elixir aims to identify the optimal combination of partitioning and offloading techniques to maximize training throughput.

Some contributions of Elixir are listed below:
- We build a pre-runtime profiler designed for large models. It is capable of obtaining the computation
graph and the memory usage of the model before training. We bring this powerful tool to support
large model profiling.
- We introduce rCache to control the degree of memory redundancy. Moreover, we build a search
engine to find the optimal configuration, maximizing training efficiency automatically. Different
from previous works, our optimal configuration considers both memory partitioning and memory
offloading.
- We conduct evaluations on a large scale by testing various model sizes, GPU capacities, numbers of
GPUs, and batch sizes. When compared to current SOTA solutions, we observe that Elixir achieves
up to 3.4× acceleration without manual tuning.

You can find more details about this system in our paper [Elixir: Train a Large Language Model on a Small GPU Cluster](https://arxiv.org/abs/2212.05339).


## 🔨 API Usage

Below is the API for the Elixir module, these APIs are experimental and subject to future changes.

### General Usage


```python
import torch
import transformers

import torch.distributed as dist

from colossalai.elixir import ElixirModule, ElixirOptimizer, minimum_waste_search

# initialize your distributed backend
...

# create your model and optimizer
model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-8)

# search for configuration
world_size = dist.get_world_size()
search_result = minimum_waste_search(model, world_size)

# wrap the model and optimizer
model = ElixirModule(model, search_result, world_group)
optimizer = ElixirOptimizer(model, optimizer)
```

### Advanced Usage

```python
import torch
import torch.distributed as dist
from colossalai.nn.optimizer import HybridAdam
from colossalai.elixir import ElixirModule, ElixirOptimizer

# initialize your distributed backend
...

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

# wrap your model with ElixirModule and optimizer with ElixirOptimizer
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
    initial_scale=1024,  # loss scale used in AMP
    init_step=True  # enable for the stability of training
)
```

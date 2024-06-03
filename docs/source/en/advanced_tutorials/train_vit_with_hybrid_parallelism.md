# Step By Step: Accelerate ViT Training With Colossal-AI (From Data Parallel to Hybrid Parallel)

Author: Yuxuan Lou, Mingyan Jiang

**Prerequisite:**
- [parallelism plugin](../basics/booster_plugins.md)
- [booster API](../basics/booster_api.md)

**Example Code**

- [Colossal-AI Examples ViT on `beans`](https://github.com/hpcaitech/ColossalAI/blob/main/examples/images/vit/vit_train_demo.py)

**Related Paper**
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)


## Introduction

In this example for ViT model, Colossal-AI provides three different parallelism techniques which accelerate model training: data parallelism, pipeline parallelism and tensor parallelism.
We will show you how to train ViT on `beans` dataset with these parallelism techniques. To run this example, you will need 2-4 GPUs.


## Table of Contents
1. Colossal-AI installation
2. Define the ViT model and related training components.
3. Boost the VIT Model with [`HybridParallelPlugin`](../basics/booster_plugins.md)
4. Train the VIT model using data parallelism, pipeline parallelism, and tensor parallelism.

## Colossal-AI Installation
You can install Colossal-AI package and its dependencies with PyPI.
```bash
pip install colossalai
```


## Import libraries
```python
from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from data import BeansDataset, beans_collator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
```
## Define the Vision Transformer (VIT) model.
Define hyperparameters.
```python
SEED = 42
MODEL_PATH = "google/vit-base-patch16-224"
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.0
NUM_EPOCH = 3
WARMUP_RATIO = 0.3
TP_SIZE = 2
PP_SIZE = 2
```
Create a distributed environment.
```python
# Launch ColossalAI
colossalai.launch_from_torch( seed=SEEDå)
coordinator = DistCoordinator()
world_size = coordinator.world_size
```
Before training, you can define the relevant components of the model training process as usual, such as defining the model, data loaders, optimizer, and so on. It's important to note that when using pipeline parallelism, you also need to define a criterion function. This function takes the input and output of the model forward pass as inputs and returns the loss.
Prepare the dataset. BeansDataset is defined in [data.py](https://github.com/hpcaitech/ColossalAI/blob/main/examples/images/vit/data.py).

```python
image_processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
train_dataset = BeansDataset(image_processor, TP_SIZE, split="train")
eval_dataset = BeansDataset(image_processor, RP_SIZE, split="validation")
num_labels = train_dataset.num_labels
```
Define the VIT model:
```python
config = ViTConfig.from_pretrained(MODEL_PATH)
config.num_labels = num_labels
config.id2label = {str(i): c for i, c in enumerate(train_dataset.label_names)}
config.label2id = {c: str(i) for i, c in enumerate(train_dataset.label_names)}
model = ViTForImageClassification.from_pretrained(
    MODEL_PATH, config=config, ignore_mismatched_sizes=True
)
```
Define the optimizer:
```python
optimizer = HybridAdam(model.parameters(), lr=(LEARNING_RATE * world_size), weight_decay=WEIGHT_DECAY)
```
Define the learning rate scheduler:
```python
total_steps = len(train_dataloader) * NUM_EPOCH
num_warmup_steps = int(WARMUP_RATIO * total_steps)
lr_scheduler = CosineAnnealingWarmupLR(
        optimizer=optimizer, total_steps=(len(train_dataloader) * NUM_EPOCH), warmup_steps=num_warmup_steps
    )
```
Define the criterion function:
```python
def _criterion(outputs, inputs):
    return outputs.loss
```
## Boost the VIT Model
We begin using ColossalAI's hybrid parallelism strategy to enhance the model. First, let's define an object of `HybridParallelPlugin`. `HybridParallelPlugin` encapsulates various parallelism strategies in ColossalAI. Afterward, we use the `HybridParallelPlugin` object to initialize the booster and boost the VIT model.
### Training with AMP
In the HybridParallelPlugin plugin, you can determine the training precision by setting the precision parameter, which supports three types: 'fp16', 'bf16', and 'fp32'. 'fp16' and 'bf16' are half-precision types. Half-precision is used in two scenarios in the HybridParallelPlugin:
1. When using zero-data parallelism, you should set it to half-precision.
2. When specifying the use of AMP (Automatic Mixed Precision) for training.
You can set related parameters when using half-precision.
`initial_scale` (float, optional): Initial loss scaling factor for AMP. Default value is 2**16.
`min_scale` (float, optional): Minimum loss scaling factor for AMP. Default value is 1.
`growth_factor` (float, optional): Multiplicative factor used to increase the loss scaling factor when using AMP. Default value is 2.
`backoff_factor` (float, optional): Multiplicative factor used to decrease the loss scaling factor when using AMP. Default value is 0.5.
`growth_interval` (integer, optional): Number of steps to increase the loss scaling factor when using AMP, in cases where there is no overflow. Default value is 1000.
`hysteresis` (integer, optional): Number of overflows required before reducing the loss scaling factor when using AMP. Default value is 2.
`max_scale` (float, optional): Maximum loss scaling factor for AMP. Default value is 2**32.
Plugin example when using amp:
```python
plugin = HybridParallelPlugin(
            precision="fp16",
            initial_scale=1,
        )
```
### Tensor parallelism
`HybridParallelPlugin` achieves tensor parallelism through Shardformer. In this plugin, you can set the `tp_size` to determine the size of tensor parallel groups. Additionally, there are multiple parameters that can be configured to optimize tensor parallelism features when using this plugin:
`enable_all_optimization` (boolean, optional): Whether to enable all optimization methods supported by Shardformer. Currently, all optimization methods include fused normalization, flash attention, and JIT. Default is False.
`enable_fused_normalization` (boolean, optional): Whether to enable fused normalization in Shardformer. Default is False.
`enable_flash_attention` (boolean, optional): Whether to enable flash attention in Shardformer. Default is False.
`enable_jit_fused` (boolean, optional): Whether to enable JIT (Just-In-Time) fusion in Shardformer. Default is False.
`enable_sequence_parallelism` (boolean): Whether to enable sequence parallelism in Shardformer. Default is False.
`enable_sequence_overlap` (boolean): Whether to enable sequence overlap in Shardformer. Default is False.
Example of a tensor parallelism plugin:
```python
plugin = HybridParallelPlugin(
            tp_size=4,
            enable_all_optimization=True
        )
```
### Pipeline Parallelism

`HybridParallelPlugin` determines the size of pipeline parallelism groups by setting `pp_size`. `num_microbatches` is used to specify the number of microbatches into which the entire batch is divided during pipeline parallelism, and `microbatch_size` can be set to define the size of these microbatches. The plugin will prioritize using `num_microbatches` to determine the microbatch configuration.
Example of a plugin for pipeline parallelism:
```python
plugin = HybridParallelPlugin(
            pp_size=4,
            num_microbatches=None,
            microbatch_size=1
        )
```
### Data Parallelism
The `HybridParallelPlugin`'s data parallelism includes both the zero-dp series and Torch DDP. When `zero_stage` is set to 0 (the default), it means using Torch DDP. Please note that Torch DDP conflicts with pipeline parallelism and cannot be used together. When `zero_stage` is set to 1, it indicates the use of the zero1 strategy. When `zero_stage` is set to 2, it implies the use of the zero2 strategy. The zero2 strategy also cannot be used together with pipeline parallelism. If you want to use zero3, please use the [`GeminiPlugin`](../basics/booster_plugins.md).
When using data parallelism with the zero series, please set the training precision to half-precision. If you haven't specified the use of zero or pipeline parallelism, and if `world_size//(tp_size*pp_size)` is greater than 1, the HybridParallelPlugin will automatically enable the Torch DDP parallel strategy for you.
Here are some related parameters for configuring Torch DDP:
`broadcast_buffers` (boolean, optional): Whether to broadcast buffers at the beginning of training when using DDP. Default is True.
`ddp_bucket_cap_mb` (integer, optional): Size of the bucket (in MB) when using DDP. Default is 25.
`find_unused_parameters` (boolean, optional): Whether to search for unused parameters when using DDP. Default is False.
`check_reduction` (boolean, optional): Whether to check the reduction operation when using DDP. Default is False.
`gradient_as_bucket_view` (boolean, optional): Whether to use gradients as bucket views when using DDP. Default is False.
`static_graph` (boolean, optional): Whether to use a static graph when using DDP. Default is False.
Example of a plugin for Torch DDP.
```python
plugin = HybridParallelPlugin(
            tp_size=2,
            pp_size=1,
            zero_stage=0,
            precision="fp16",
            initial_scale=1,
        )
```
If there are 4 parallel processes, the parallel group size for Torch DDP is 2.
ZeRO-related parameters:
`zero_bucket_size_in_m` (integer, optional): The bucket size for gradient reduction in megabytes when using ZeRO. Default is 12.
`cpu_offload` (boolean, optional): Whether to enable cpu_offload when using ZeRO. Default is False.
`communication_dtype` (torch data type, optional): The data type for communication when using ZeRO. If not specified, the data type of the parameters will be used. Default is None.
`overlap_communication` (boolean, optional): Whether to overlap communication and computation when using ZeRO. Default is True.
Example of a plugin for ZERO1.
```python
plugin = HybridParallelPlugin(
            tp_size=1,
            pp_size=1,
            zero_stage=1,
            cpu_offload=True,
            precision="fp16",
            initial_scale=1,
        )
```

### Hybrid Parallelism
You can refer to the above-mentioned strategies to customize an appropriate hybrid parallelism strategy. And use this plugin to define a booster.
```python
plugin = HybridParallelPlugin(
            tp_size=TP_SIZE,
            pp_size=PP_SIZE,
            num_microbatches=None,
            microbatch_size=1,
            enable_all_optimization=True,
            precision="fp16",
            initial_scale=1,
        )
booster = Booster(plugin=plugin)
```
Next, we use `booster.boost` to inject the features encapsulated by the plugin into the model training components.
```python
model, optimizer, _criterion, train_dataloader, lr_scheduler = booster.boost(
        model=model, optimizer=optimizer, criterion=criterion, dataloader=train_dataloader, lr_scheduler=lr_scheduler
    )
```
## Train ViT using hybrid parallelism.
Finally, we can use the hybrid parallelism strategy to train the model. Let's first define a training function that describes the training process. It's important to note that if the pipeline parallelism strategy is used, you should call `booster.execute_pipeline` to perform the model training. This function will invoke the `scheduler` to manage the model's forward and backward operations.
```python
def run_forward_backward(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: Callable[[Any, Any], torch.Tensor],
    data_iter: Iterator,
    booster: Booster,
):
    if optimizer is not None:
        optimizer.zero_grad()
    if isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1:
        # run pipeline forward backward when enabling pp in hybrid parallel plugin
        output_dict = booster.execute_pipeline(
            data_iter, model, criterion, optimizer, return_loss=True
        )
        loss, outputs = output_dict["loss"], output_dict["outputs"]
    else:
        batch = next(data_iter)
        batch = move_to_cuda(batch, torch.cuda.current_device())
        outputs = model(**batch)
        loss = criterion(outputs, None)
        if optimizer is not None:
            booster.backward(loss, optimizer)

def train_epoch(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    criterion: Callable[[Any, Any], torch.Tensor],
    lr_scheduler: LRScheduler,
    dataloader: DataLoader,
    booster: Booster,
    coordinator: DistCoordinator,
):
    torch.cuda.synchronize()

    num_steps = len(dataloader)
    data_iter = iter(dataloader)
    enable_pbar = coordinator.is_master()
    if isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1:
        # when using pp, only the last stage of master pipeline (dp_rank and tp_rank are both zero) shows pbar
        tp_rank = dist.get_rank(booster.plugin.tp_group)
        dp_rank = dist.get_rank(booster.plugin.dp_group)
        enable_pbar = tp_rank == 0 and dp_rank == 0 and booster.plugin.stage_manager.is_last_stage()
    model.train()

    with tqdm(range(num_steps), desc=f"Epoch [{epoch + 1}]", disable=not enable_pbar) as pbar:
        for _ in pbar:
            loss, _ = run_forward_backward(model, optimizer, criterion, data_iter, booster)
            optimizer.step()
            lr_scheduler.step()

            # Print batch loss
            if enable_pbar:
                pbar.set_postfix({"loss": loss.item()})
```
Start training the model.
```python
for epoch in range(NUM_EPOCH):
    train_epoch(epoch, model, optimizer, criterion, lr_scheduler, train_dataloader, booster, coordinator)
```
<!-- doc-test-command: echo  -->

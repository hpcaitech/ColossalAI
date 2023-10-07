# Fine-tune GPT-2 Using Hybrid Parallelism

Author: Hongxin Liu, Yongbin Li, Mingyan Jiang

**Example Code**
- [ColossalAI-Examples GPT](https://github.com/flybird11111/ColossalAI/tree/main/examples/language/gpt/hybridparallelism)


**Related Paper**
- [Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training](https://arxiv.org/abs/2110.14883)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

## Introduction

In the previous tutorial, we introduce how to train ViT with pipeline. In this tutorial, you will learn a more complex scenario -- fine-tune GPT-2 with hybrid parallelism. In this case, GPT-2 is so large that CPU memory cannot fit it as well. Therefore, you must split the model.

## Table of content

In this tutorial we will cover:

1. Defining the Training Components of the GPT-2 Model
2. Boost the Training Components with [`HybridParallelPlugin`](../basics/booster_plugins.md)
3. Training GPT-2 using hybrid parallelism

## Import libraries

```python
import argparse
from typing import Callable, List, Union

import evaluate
import torch
import torch.distributed as dist
import torch.nn as nn
from data import GLUEDataBuilder
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, GPT2ForSequenceClassification, get_linear_schedule_with_warmup

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
```



## Define GPT-2‘s Training Components

Before using mixed parallelism, you can define the components needed for training according to the normal workflow.

Define hyperparameters
```python
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2.4e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
```
Prepare dataloader
```python
data_builder = GLUEDataBuilder(
    model_name, plugin, args.task, train_batch_size=BATCH_SIZE, eval_batch_size=BATCH_SIZE
)
train_dataloader = data_builder.train_dataloader()
test_dataloader = data_builder.test_dataloader()
```
Prepare gpt-2 model
```python
cfg = AutoConfig.from_pretrained(model_name, num_labels=data_builder.num_labels)

if model_name == "gpt2":
    model = GPT2ForSequenceClassification.from_pretrained(model_name, config=cfg).cuda()
else:
    raise RuntimeError
```
Prepare optimizer and criterion, It's important to note that, during hybrid parallelism training, a callable function variable should be passed the `execute_pipeline`. This function should take 'input' and 'output' as parameters and return the loss.
```python
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": WEIGHT_DECAY,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = HybridAdam(optimizer_grouped_parameters, lr=lr, eps=1e-8)
```
Prepare lr_scheduler and criterion
```python
output_transform_fn = lambda x: x
criterion = lambda x: x.loss
# lr scheduler
total_steps = len(train_dataloader) * NUM_EPOCHS
num_warmup_steps = int(WARMUP_FRACTION * total_steps)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=total_steps,
)

def _criterion(outputs, inputs):
    outputs = output_transform_fn(outputs)
    loss = criterion(outputs)
    return loss
```

Define a booster with `HybridParallelPlugin`. Based on the configured plugin parameters, the booster will inject one or more parallel strategies into the model. In this example, pipeline parallelism, zero1, and mixed-precision training optimizations are utilized.
```python
plugin = HybridParallelPlugin(
    tp_size=1,
    pp_size=2,
    num_microbatches=None,
    microbatch_size=1,
    enable_all_optimization=True,
    zero_stage=1,
    precision="fp16",
    initial_scale=1,
)

booster = Booster(plugin=plugin, **booster_kwargs)
```
Boost these components with defined booster
```python
model, optimizer, _criterion, _, lr_scheduler = booster.boost(
    model, optimizer, criterion=_criterion, lr_scheduler=lr_scheduler
)
```


## Training GPT-2 using hybrid parallelism

In the previous tutorial, We've explained how to inject various parallelism features into the model and its training components using the Booster and `HybridParallelPlugin`. Now we can start model training.
Define a training function
```python
def train_epoch(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    _criterion: Callable,
    lr_scheduler: LRScheduler,
    train_dataloader: DataLoader,
    booster: Booster,
    coordinator: DistCoordinator,
):
    is_pp_last_stage = booster.plugin.stage_manager.is_last_stage()
    total_step = len(train_dataloader)

    model.train()
    optimizer.zero_grad()
    train_dataloader_iter = iter(train_dataloader)
    with tqdm(
        range(total_step),
        desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]",
        disable=not (coordinator.is_master() or is_pp_last_stage),
    ) as pbar:
        # Forward pass
        for _ in pbar:
            outputs = booster.execute_pipeline(
                train_dataloader_iter, model, _criterion, optimizer, return_loss=True, return_outputs=True
            )
            # Backward and optimize
            if is_pp_last_stage:
                loss = outputs["loss"]
                pbar.set_postfix({"loss": loss.item()})

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
```
Training the gpt-2 model
```python
for epoch in range(NUM_EPOCHS):
    train_epoch(epoch, model, optimizer, _criterion, lr_scheduler, train_dataloader, booster, coordinator)
```
<!-- doc-test-command: echo  -->
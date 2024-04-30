# Fine-tune GPT-2 Using Hybrid Parallelism

Author: Hongxin Liu, Yongbin Li, Mingyan Jiang

**Prerequisite:**
- [parallelism plugin](../basics/booster_plugins.md)
- [booster API](../basics/booster_api.md)

**Example Code**
- [ColossalAI-Examples GPT](https://github.com/hpcaitech/ColossalAI/blob/main/examples/language/gpt/hybridparallelism/finetune.py)


**Related Paper**
- [Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training](https://arxiv.org/abs/2110.14883)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

## Introduction

In the previous tutorial, we introduce how to train ViT with pipeline. In this tutorial, you will learn a more complex scenario -- fine-tune GPT-2 with hybrid parallelism. In this case, GPT-2 is so large that CPU memory cannot fit it as well. Therefore, you must split the model.

## Table of content

In this tutorial we will cover:

1. Initialize the hybrid parallelism plugin.
2. Defining the Training Components of the GPT-2 Model
3. Boost the GPT-2 Model with [`HybridParallelPlugin`](../basics/booster_plugins.md)
4. Training GPT-2 using hybrid parallelism

## Import libraries

```python
from typing import Callable, List, Union
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from tqdm import tqdm
from transformers import AutoConfig, GPT2ForSequenceClassification, get_linear_schedule_with_warmup
from transformers import AutoTokenizer

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
```
## Define Plugin
Create a `HybridParallelPlugin` object and specify the desired parallelism strategies to be used. In this example, both pipeline parallelism and ZeRO-1 are used simultaneously.
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
```
## Define GPT-2's Training Components

Before using hybrid parallelism, you need to define the components used for training.

Define hyperparameters
```python
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2.4e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
```
we create a distributed environment.
```python
# Launch ColossalAI
colossalai.launch_from_torch( seed=42)
coordinator = DistCoordinator()
```
prepare the dataset. You can use `plugin.prepare_dataloader` to generate a dataloader or customize your own dataloader.
```python
def tokenize_batch(batch, tokenizer: Optional[AutoTokenizer] = None, max_length: int = 2048):
    texts = [sample["sentence1"] + sample["sentence2"] for sample in batch]
    data = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    data = {k: v.cuda() for k, v in data.items()}
    data["labels"] = data["input_ids"].clone()
    return data

tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = datasets.load_dataset("glue", "mrpc")
train_dataloader = plugin.prepare_dataloader(
    dataset["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    collate_fn=partial(tokenize_batch, tokenizer=tokenizer, max_length=512),
)
```
Prepare gpt-2 model
```python
cfg = AutoConfig.from_pretrained("gpt2", num_labels=2)
model = GPT2ForSequenceClassification.from_pretrained("gpt2", config=cfg).cuda()

```
prepare optimizer
```python
lr = LEARNING_RATE * coordinator.world_size
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
Prepare the lr_scheduler and criterion, and it's important to note that when hybrid parallelism with pipeline parallelism is used, a criterion function should also be defined. This function should take the input and output of the model's forward pass as parameters and return the loss.
```python
# lr scheduler
total_steps = len(train_dataloader) * NUM_EPOCHS
num_warmup_steps = int(WARMUP_FRACTION * total_steps)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=total_steps,
)

def _criterion(outputs, inputs):
    return outputs.loss
```
## Boost the GPT-2 Model
Define a booster with `HybridParallelPlugin`. Based on the configured plugin parameters, the booster will inject one or more parallel strategies into the model. In this example, pipeline parallelism, zero1, and mixed-precision training optimizations are utilized.
```python
booster = Booster(plugin=plugin)
```
Boost these components with the defined booster.
```python
model, optimizer, _criterion, _, lr_scheduler = booster.boost(
    model, optimizer, criterion=_criterion, lr_scheduler=lr_scheduler
)
```


## Training GPT-2 using hybrid parallelism

In the previous tutorial, We've explained how to inject various parallelism features into the model and its training components using the Booster and `HybridParallelPlugin`. Now we can start model training.
Define a training function. When pipeline parallelism is used, you need to call `booster.execute_pipeline` to schedule the stages of model training.
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
    use_pipeline = isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1
    is_pp_last_stage = use_pipeline and booster.plugin.stage_manager.is_last_stage()
    print_flag = (not use_pipeline and coordinator.is_master()) or (use_pipeline and is_pp_last_stage)
    total_step = len(train_dataloader)

    model.train()
    optimizer.zero_grad()
    train_dataloader_iter = iter(train_dataloader)
    with tqdm(
        range(total_step),
        desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]",
        disable=not print_flag,
    ) as pbar:
        # Forward pass
        for _ in pbar:
            if use_pipeline:
                outputs = booster.execute_pipeline(
                    train_dataloader_iter, model, _criterion, optimizer, return_loss=True
                )
                # Backward and optimize
                if is_pp_last_stage:
                    loss = outputs["loss"]
                    pbar.set_postfix({"loss": loss.item()})
            else:
                data = next(train_dataloader_iter)
                data = move_to_cuda(data)
                outputs = model(**data)
                loss = _criterion(outputs, None)
                # Backward
                booster.backward(loss, optimizer)
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
<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 train_gpt_using_hybrid_parallelism.py  -->

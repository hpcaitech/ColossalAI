# Train ViT Using Pipeline Parallelism

Author: Hongxin Liu, Yongbin Li

**Prerequisite:**
- [parallellism plugin](../basics/booster_plugins.md)
- [booster API](../basics/booster_api.md)

**Example Code**
- [ColossalAI-Examples Pipeline Parallel ViT](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/vit)

**Related Paper**
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

## Introduction

In this tutorial, you will learn how to train Vision Transformer for image classification from scratch, using pipeline.
Pipeline parallelism is a kind of model parallelism, which is useful when your GPU memory cannot fit your model.
By using it, we split the original model into multi stages, and each stage maintains a part of the original model.
We assume that your GPU memory cannot fit ViT/L-16, and your memory can fit this model.

##  Table of contents

In this tutorial we will cover:

1. Define the ViT model and related training components.
2. Boost the VIT Model with [`HybridParallelPlugin`](../basics/booster_plugins.md)
3. Train the ViT model using pipeline parallelism.

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
## Define Vision Transformer model
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
First, we create a distributed environment.
```python
    # Launch ColossalAI
    colossalai.launch_from_torch(config={}, seed=SEED)
    coordinator = DistCoordinator()
    world_size = coordinator.world_size
```
Before training, you can define the relevant components for model training according to the normal workflow, such as defining the model, data loaders, optimizers, and so on. It's important to note that when using pipeline parallelism, you also need to define a criterion function. This function takes the input and output of the model's forward pass as inputs and returns the loss.
Define the model:
```python
    config = ViTConfig.from_pretrained(MODEL_PATH)
    config.num_labels = num_labels
    config.id2label = {str(i): c for i, c in enumerate(train_dataset.label_names)}
    config.label2id = {c: str(i) for i, c in enumerate(train_dataset.label_names)}
    model = ViTForImageClassification.from_pretrained(
        MODEL_PATH, config=config, ignore_mismatched_sizes=True
    )
```
Define optimizer：
```python
optimizer = HybridAdam(model.parameters(), lr=(LEARNING_RATE * world_size), weight_decay=WEIGHT_DECAY)
```
Define the lr scheduler:
```python
total_steps = len(train_dataloader) * NUM_EPOCH
num_warmup_steps = int(WARMUP_RATIO * total_steps)
lr_scheduler = CosineAnnealingWarmupLR(
        optimizer=optimizer, total_steps=(len(train_dataloader) * total_steps = len(train_dataloader) * NUM_EPOCH
), warmup_steps=num_warmup_steps
    )
```
Generally, we train ViT on large dataset like Imagenet. For simplicity, we just use CIFAR-10 here, since this tutorial is just for pipeline training.

```python
def build_cifar(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(224, pad_if_needed=True),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR10(root=os.environ['DATA'], train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root=os.environ['DATA'], train=False, transform=transform_test)
    train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
    test_dataloader = get_dataloader(dataset=test_dataset, batch_size=batch_size, pin_memory=True)
    return train_dataloader, test_dataloader
```
prepare the dataset.
```python
image_processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
train_dataset = BeansDataset(image_processor, TP_SIZE, split="train")
eval_dataset = BeansDataset(image_processor, TP_SIZE, split="validation")
num_labels = train_dataset.num_labels
```
Define the criterion function:
```python
def _criterion(outputs, inputs):
    outputs = output_transform_fn(outputs)
    loss = criterion(outputs)
    return loss
```
## Boost VIT Model
We begin by enhancing the model with colossalai's pipeline parallelism strategy. First, we define a HybridParallelPlugin object. HybridParallelPlugin encapsulates various parallelism strategies in colossalai. You can specify the use of pipeline parallelism by setting three parameters: pp_size, num_microbatches, and microbatch_size. For specific parameter settings, refer to the plugin-related documentation. Then, we initialize the booster with the HybridParallelPlugin object.
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
booster = Booster(plugin=plugin, **booster_kwargs)
```
Next, we use booster.boost to inject the features encapsulated by the plugin into the model training components.
```python
model, optimizer, _criterion, train_dataloader, lr_scheduler = booster.boost(
        model=model, optimizer=optimizer, criterion=criterion, dataloader=train_dataloader, lr_scheduler=lr_scheduler
    )
```
## Training ViT using pipeline
Finally, we can train the model using pipeline parallelism. First, we define a training function that describes the training process. It's important to note that when using pipeline parallelism, you need to call booster.execute_pipeline to perform the model training. This function will invoke the scheduler to manage the model's forward and backward operations.
```python
def run_forward_backward(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: Callable[[Any, Any], torch.Tensor],
    data_iter: Iterator,
    booster: Booster,
):
# run pipeline forward backward when enabling pp in hybrid parallel plugin
output_dict = booster.execute_pipeline(
    data_iter, model, criterion, optimizer, return_loss=True, return_outputs=True
)
loss, outputs = output_dict["loss"], output_dict["outputs"]


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

# 使用流水并行训练 ViT

作者: Hongxin Liu, Yongbin Li, Mingyan Jiang

**前置教程**
- [并行插件](../basics/booster_plugins.md)
- [booster API](../basics/booster_api.md)

**示例代码**
- [ColossalAI-Examples Pipeline Parallel ViT](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/vit)

**相关论文**
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

## 引言

在本教程中，你将学习如何使用流水并行从头开始训练用于图像分类的 Vision Transformer (ViT)。流水并行是一种模型并行，主要针对 GPU 内存不能满足模型容量的情况。
通过使用流水并行，我们将原始模型分割成多个阶段，每个阶段保留原始模型的一部分。我们假设你的 GPU 内存不能容纳 ViT/L-16，而你的内存可以容纳这个模型。

##  目录

在本教程中，我们将介绍:

1. 定义ViT模型及相关训练组件
2. 使用 [HybridParallelPlugin](../basics/booster_plugins.md) 增强VIT模型
3. 使用流水并行训练 ViT

## 导入依赖库

```python
from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from args import parse_demo_args
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
## 定义 Vision Transformer 模型
首先我们创建一个分布式环境
```python
    # Launch ColossalAI
    colossalai.launch_from_torch(config={}, seed=args.seed)
    coordinator = DistCoordinator()
    world_size = coordinator.world_size
```
在训练之前您可以按照正常流程定义模型训练的相关组，如定义模型，数据加载器，优化器等。需要注意的是，当使用管道并行时，还需定义一个criterion函数，该函数的输入是模型前向的输入和输出，返回的是loss。
定义模型：
```python
    config = ViTConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = num_labels
    config.id2label = {str(i): c for i, c in enumerate(train_dataset.label_names)}
    config.label2id = {c: str(i) for i, c in enumerate(train_dataset.label_names)}
    model = ViTForImageClassification.from_pretrained(
        args.model_name_or_path, config=config, ignore_mismatched_sizes=True
    )
```
定义optimizer：
```python
optimizer = HybridAdam(model.parameters(), lr=(args.learning_rate * world_size), weight_decay=args.weight_decay)
```
定义lr scheduler
```python
total_steps = len(train_dataloader) * args.num_epoch
num_warmup_steps = int(args.warmup_ratio * total_steps)
lr_scheduler = CosineAnnealingWarmupLR(
        optimizer=optimizer, total_steps=(len(train_dataloader) * args.num_epoch), warmup_steps=num_warmup_steps
    )
```
一般来说, 我们在大型数据集如 ImageNet 上训练 ViT。为了简单期间，我们在这里只使用 CIFAR-10, 因为本教程只是用于流水并行训练。

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
定义criterion函数：
```python
def _criterion(outputs, inputs):
    outputs = output_transform_fn(outputs)
    loss = criterion(outputs)
    return loss
```
获取数据集
```python
image_processor = ViTImageProcessor.from_pretrained(args.model_name_or_path)
train_dataset = BeansDataset(image_processor, args.tp_size, split="train")
eval_dataset = BeansDataset(image_processor, args.tp_size, split="validation")
num_labels = train_dataset.num_labels
```
## 增强VIT模型
我们开始使用colossalai的管道并行策略来增强模型，首先我们先定义一个`HybridParallelPlugin`的对象，[`HybridParallelPlugin`](../basics/booster_plugins.md)封装了colossalai的多种并行策略，通过设置`pp_size`、`num_microbatches`、`microbatch_size`这三个参数可来指定使用管道并行策略，具体参数设置可参考plugin相关文档。之后我们使用`HybridParallelPlugin`对象来初始化booster。
```python
plugin = HybridParallelPlugin(
            tp_size=args.tp_size,
            pp_size=args.pp_size,
            num_microbatches=None,
            microbatch_size=1,
            enable_all_optimization=True,
            precision="fp16",
            initial_scale=1,
        )
booster = Booster(plugin=plugin, **booster_kwargs)
```
接着我们使用`booster.boost`来将plugin所封装的特性注入到模型训练组件中。
```python
model, optimizer, _criterion, train_dataloader, lr_scheduler = booster.boost(
        model=model, optimizer=optimizer, criterion=criterion, dataloader=train_dataloader, lr_scheduler=lr_scheduler
    )
```
## 使用流水并行训练 ViT
最后我们就可以使用管道并行来训练模型了，我们先定义一个训练函数，描述训练过程，需要注意的是，如果使用了管道并行策略，需要调用`booster.execute_pipeline`来执行模型的训练，它会调用`scheduler`管理模型的前后向操作。
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
开始训练模型
```python
for epoch in range(args.num_epoch):
    train_epoch(epoch, model, optimizer, criterion, lr_scheduler, train_dataloader, booster, coordinator)
```
训练完成后，可调用`booster.save_model`保存模型。
```python
booster.save_model(model, args.output_path, shard=True)
```
<!-- doc-test-command: echo  -->

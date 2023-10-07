# 使用混合并行训练 GPT-2

作者: Hongxin Liu, Yongbin Li, Mingyan Jiang

**示例代码**
- [ColossalAI-Examples GPT2](https://github.com/flybird11111/ColossalAI/tree/main/examples/language/gpt/hybridparallelism)

**相关论文**
- [Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training](https://arxiv.org/abs/2110.14883)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

## 引言

在上一篇教程中，我们介绍了如何用流水并行训练 ViT。在本教程中，你将学习一个更复杂的场景--用混合并行方式训练GPT-2。在这种情况下，由于GPT-2过大，即使CPU内存也无法容纳它。因此，该模型必须被分割。

## 目录

在本教程中，我们将介绍:

1. 定义 GPT-2 模型的训练组件
2. 使用 [HybridParallelPlugin](../basics/booster_plugins.md) 增强训练组件
3. 使用混合并行训练 GPT-2

## 导入依赖库

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

## 定义GPT-2模型的训练组件

在使用混合并行之前，您可以按照正常流程定义训练所需的组件。

定义超参数。
```python
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2.4e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
```
定义数据加载器。
```python
data_builder = GLUEDataBuilder(
    model_name, plugin, args.task, train_batch_size=BATCH_SIZE, eval_batch_size=BATCH_SIZE
)
train_dataloader = data_builder.train_dataloader()
test_dataloader = data_builder.test_dataloader()
```
定义GPT-2模型。
```python
cfg = AutoConfig.from_pretrained(model_name, num_labels=data_builder.num_labels)

if model_name == "gpt2":
    model = GPT2ForSequenceClassification.from_pretrained(model_name, config=cfg).cuda()
else:
    raise RuntimeError
```
准备优化器和损失函数。需要注意的是，在混合并行训练期间，应该传递一个可调用的函数变量给`execute_pipeline`。这个函数应该以 'input' 和 'output' 作为参数，并返回损失值。
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

使用 HybridParallelPlugin 定义一个 booster（增强器）。根据设置的插件参数，booster会将一种或者多种并行策略注入到模型中。该例子中使用了管道并行，zero1，及半精度训练等优化。
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
使用定义的 booster 来增强这些组件。
```python
model, optimizer, _criterion, _, lr_scheduler = booster.boost(
    model, optimizer, criterion=_criterion, lr_scheduler=lr_scheduler
)
```


## 使用混合并行训练 GPT-2

在前面的教程中，我们已经解释了如何使用 Booster 和 HybridParallelPlugin 将各种并行特性注入到模型及其训练组件中。现在我们可以开始模型训练。
定义一个训练函数。
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
            if use_pipeline:
                outputs = booster.execute_pipeline(
                    train_dataloader_iter, model, _criterion, optimizer, return_loss=True, return_outputs=True
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
训练 GPT-2 模型。
```python
for epoch in range(NUM_EPOCHS):
    train_epoch(epoch, model, optimizer, _criterion, lr_scheduler, train_dataloader, booster, coordinator)
```
<!-- doc-test-command: echo  -->
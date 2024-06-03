# 使用混合并行训练 GPT-2

作者: Hongxin Liu, Yongbin Li, Mingyan Jiang

**前置教程**
- [并行插件](../basics/booster_plugins.md)
- [booster API](../basics/booster_api.md)

**示例代码**
- [ColossalAI-Examples GPT2](https://github.com/hpcaitech/ColossalAI/blob/main/examples/language/gpt/hybridparallelism/finetune.py)

**相关论文**
- [Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training](https://arxiv.org/abs/2110.14883)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

## 引言

在上一篇教程中，我们介绍了如何用流水并行训练 ViT。在本教程中，你将学习一个更复杂的场景--用混合并行方式训练GPT-2。在这种情况下，由于GPT-2过大，即使CPU内存也无法容纳它。因此，该模型必须被分割。

## 目录

在本教程中，我们将介绍:
1. 初始化混合并行插件
2. 定义 GPT-2 模型的训练组件
3. 使用 [HybridParallelPlugin](../basics/booster_plugins.md) 增强GPT-2模型
4. 使用混合并行训练 GPT-2

## 导入依赖库

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
### 定义plugin
定义一个[`HybridParallelPlugin`](../basics/booster_plugins.md)对象，指定所需要使用的并行策略，在该例子中，同时使用了流水线并行和zero1.
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

## 创建分布式环境.
```python
# Launch ColossalAI
colossalai.launch_from_torch(seed=42)
coordinator = DistCoordinator()
```
## 定义GPT-2模型的训练组件
在使用混合并行之前，您需要定义训练所使用的组件。
定义超参数。
```python
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2.4e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
```
获取数据集。您可以使用`plugin.prepare_dataloader`生成dataloader,也可以自定义您的dataloader。
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
定义GPT-2模型。
```python
cfg = AutoConfig.from_pretrained("gpt2", num_labels=2)
model = GPT2ForSequenceClassification.from_pretrained("gpt2", config=cfg).cuda()
```
准备优化器
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
准备 `lr_scheduler` 和 `criterion`，需要注意的是，当混合并行使用了管道并行时，还需定义`criterion`函数。这个函数应该以模型前后向的输入和输出作为参数，并返回loss。
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
## 增强GPT-2模型
使用 HybridParallelPlugin 定义一个 booster（增强器）。根据设置的插件参数，booster会将一种或者多种并行策略注入到模型中。该例子中使用了管道并行，zero1，及半精度训练等优化。
```python
booster = Booster(plugin=plugin)
```
使用定义的 booster 来增强这些组件。
```python
model, optimizer, _criterion, _, lr_scheduler = booster.boost(
    model, optimizer, criterion=_criterion, lr_scheduler=lr_scheduler
)
```


## 使用混合并行训练 GPT-2

在前面的教程中，我们已经解释了如何使用 Booster 和 HybridParallelPlugin 将各种并行特性注入到模型及其训练组件中。现在我们可以开始模型训练。
定义一个训练函数。当使用了管道并行时，需要调用`booster.execute_pipeline`进行模型训练的阶段调度。
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
训练 GPT-2 模型。
```python
for epoch in range(NUM_EPOCHS):
    train_epoch(epoch, model, optimizer, _criterion, lr_scheduler, train_dataloader, booster, coordinator)
```
<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 train_gpt_using_hybrid_parallelism.py  -->

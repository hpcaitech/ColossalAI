# 使用 Colossal-AI （从数据并行到异构并行）加速 ViT 训练详解

作者：Yuxuan Lou, Mingyan Jiang

**前置教程**
- [并行插件](../basics/booster_plugins.md)
- [booster API](../basics/booster_api.md)

**示例代码**

- [Colossal-AI Examples ViT on `beans`](https://github.com/hpcaitech/ColossalAI/blob/main/examples/images/vit/vit_train_demo.py)

**相关文献**
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)


## 引言

在这个ViT模型的样例中，Colossal-AI 提供了三种不同的并行技术来加速模型训练：数据并行，流水线并行和张量并行。我们将展示如何使用这三种并行技术在 `beans` 数据集上训练 ViT。为了运行项目，需要2-4个 GPU。


## 目录
1. Colossal-AI 安装方法
2. 定义VIT模型及相关训练组件
3. 使用使用 [HybridParallelPlugin](../basics/booster_plugins.md) 增强VIT模型
4. 使用数据并行、流水线并行及张量并行训练VIT模型

## Colossal-AI 安装
可以通过 Python 的官方索引来安装 Colossal-AI 软件包。
```bash
pip install colossalai
```

## 导入依赖库

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
## 定义 Vision Transformer 模型
定义超参数
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
首先我们创建一个分布式环境
```python
# Launch ColossalAI
colossalai.launch_from_torch(seed=SEEDå)
coordinator = DistCoordinator()
world_size = coordinator.world_size
```
在训练之前您可以按照正常流程定义模型训练的相关组，如定义模型，数据加载器，优化器等。需要注意的是，当使用管道并行时，还需定义一个criterion函数，该函数的输入是模型前向的输入和输出，返回的是loss。
获取数据集, `BeansDataset`定义在[data.py](https://github.com/hpcaitech/ColossalAI/blob/main/examples/images/vit/data.py)
```python
image_processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
train_dataset = BeansDataset(image_processor, TP_SIZE, split="train")
eval_dataset = BeansDataset(image_processor, RP_SIZE, split="validation")
num_labels = train_dataset.num_labels
```
定义VIT模型：
```python
config = ViTConfig.from_pretrained(MODEL_PATH)
config.num_labels = num_labels
config.id2label = {str(i): c for i, c in enumerate(train_dataset.label_names)}
config.label2id = {c: str(i) for i, c in enumerate(train_dataset.label_names)}
model = ViTForImageClassification.from_pretrained(
    MODEL_PATH, config=config, ignore_mismatched_sizes=True
)
```
定义optimizer：
```python
optimizer = HybridAdam(model.parameters(), lr=(LEARNING_RATE * world_size), weight_decay=WEIGHT_DECAY)
```
定义lr scheduler:
```python
total_steps = len(train_dataloader) * NUM_EPOCH
num_warmup_steps = int(WARMUP_RATIO * total_steps)
lr_scheduler = CosineAnnealingWarmupLR(
        optimizer=optimizer, total_steps=(len(train_dataloader) * NUM_EPOCH), warmup_steps=num_warmup_steps
    )
```
定义criterion函数：
```python
def _criterion(outputs, inputs):
    return outputs.loss
```
## 增强VIT模型
我们开始使用colossalai的混合并行策略来增强模型，首先我们先定义一个`HybridParallelPlugin`的对象，[`HybridParallelPlugin`](../basics/booster_plugins.md)封装了colossalai的多种并行策略，之后我们使用`HybridParallelPlugin`对象来初始化booster并调用`booster.boost`来增强模型。
### 半精度训练
在`HybridParallelPlugin`插件中，通过设置`precision`确定训练精度，可支持'fp16','bf16','fp32'三种类型。'fp16','bf16'为半精度类型，半精度在`HybridParallelPlugin`中有两种应用场景，一是使用zero数据并行时，需设置为半精度；二是指定使用amp半精度进行训练。

使用amp半精度时，可设置相关参数。
`initial_scale`（浮点数，可选项）：AMP的初始损失缩放比例。默认值为2**16。
`min_scale`（浮点数，可选项）：AMP的最小损失缩放比例。默认值为1。
`growth_factor`（浮点数，可选项）：在使用AMP时，用于增加损失缩放比例的乘法因子。默认值为2。
`backoff_factor`（浮点数，可选项）：在使用AMP时，用于减少损失缩放比例的乘法因子。默认值为0.5。
`growth_interval`（整数，可选项）：在使用AMP时，当没有溢出时增加损失缩放比例的步数。默认值为1000。
`hysteresis`（整数，可选项）：在使用AMP时，减少损失缩放比例之前的溢出次数。默认值为2。
`max_scale`（浮点数，可选项）：AMP的最大损失缩放比例。默认值为2**32。

使用AMP的plugin示例：
```python
plugin = HybridParallelPlugin(
            precision="fp16",
            initial_scale=1,
        )
```

### 张量并行
`HybridParallelPlugin`是通过shardformer实现张量并行，在该插件中，可设置`tp_size`确定张量并行组的大小，此外，还有多个参数可设置张量并行时的优化特性：

`enable_all_optimization`（布尔类型，可选项）：是否启用Shardformer支持的所有优化方法，目前所有优化方法包括融合归一化、flash attention和JIT。默认为False。
`enable_fused_normalization`（布尔类型，可选项）：是否在Shardformer中启用融合归一化。默认为False。
`enable_flash_attention`（布尔类型，可选项）：是否在Shardformer中启用flash attention。默认为False。
`enable_jit_fused`（布尔类型，可选项）：是否在Shardformer中启用JIT。默认为False。
`enable_sequence_parallelism`（布尔类型）：是否在Shardformer中启用序列并行性。默认为False。
`enable_sequence_overlap`（布尔类型）：是否在Shardformer中启用序列重叠性。默认为False。

张量并行的plugin示例
```python
plugin = HybridParallelPlugin(
            tp_size=4,
            enable_all_optimization=True
        )
```
### 流水线并行
`HybridParallelPlugin`通过设置`pp_size`确定流水线并行组的大小，`num_microbatches`设置流水线并行时将整个batch划分为小batch的数量，`microbatch_size`可设置小batch的大小，插件会优先使用`num_microbatches`来确定micro batch的配置。
流水线并行的plugin示例
```python
plugin = HybridParallelPlugin(
            pp_size=4,
            num_microbatches=None,
            microbatch_size=1
        )
```
### 数据并行
`HybridParallelPlugin`插件的数据并行包括zero-dp系列及torch DDP。当`zero_stage`为0(默认值)时表示使用torch DDP，注意torch DDP与流水线并行有冲突，不能一起使用。`zero_stage`为1时表示使用zero1策略。`zero_stage`为2使用zero2,zero2策略也无法与流水线并行一起使用。如果想使用zero3，请使用[`GeminiPlugin`](../basics/booster_plugins.md)。使用zero系列的数据并行，请设置训练精度为半精度。当未指定使用zero及流水线并行，且world_size//(tp_size*pp_size)大于1时，`HybridParallelPlugin`会为您打开torch DDP并行策略。
torch DDP相关参数设置：
`broadcast_buffers`（布尔值，可选项）：在使用DDP时，在训练开始时是否广播缓冲区。默认为True。
`ddp_bucket_cap_mb`（整数，可选项）：在使用DDP时的桶大小（以MB为单位）。默认为25。
`find_unused_parameters`（布尔值，可选项）：在使用DDP时是否查找未使用的参数。默认为False。
`check_reduction（布尔值，可选项）：在使用DDP时是否检查减少。默认为False。
`gradient_as_bucket_view`（布尔值，可选项）：在使用DDP时是否将梯度作为桶视图使用。默认为False。
`static_graph`（布尔值，可选项）：在使用DDP时是否使用静态图。默认为False。

Torch DDP的plugin示例
```python
plugin = HybridParallelPlugin(
            tp_size=2,
            pp_size=1,
            zero_stage=0,
            precision="fp16",
            initial_scale=1,
        )
```
若并行进程为4，则torch DDP的并行组大小为2.
zero相关参数设置：
`zero_bucket_size_in_m`（整数，可选项）：在使用ZeRO时，以百万元素为单位的梯度减小桶大小。默认为12。
`cpu_offload`（布尔值，可选项）：在使用ZeRO时是否打开`cpu_offload`。默认为False。
`communication_dtype`（torch数据类型，可选项）：在使用ZeRO时的通信数据类型。如果未指定，则将使用参数的数据类型。默认为None。
`overlap_communication`（布尔值，可选项）：在使用ZeRO时是否重叠通信和计算。默认为True。

zero1的plugin示例

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

### 混合并行
可参考上述的策略自定义合适的混合并行策略。定义混合并行的插件，并使用该插件定义一个booster：

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
接着我们使用`booster.boost`来将plugin所封装的特性注入到模型训练组件中。
```python
model, optimizer, _criterion, train_dataloader, lr_scheduler = booster.boost(
        model=model, optimizer=optimizer, criterion=criterion, dataloader=train_dataloader, lr_scheduler=lr_scheduler
    )
```
## 使用混合并行训练 ViT
最后就可以使用混合并行策略来训练模型了。我们先定义一个训练函数，描述训练过程。需要注意的是，如果使用了管道并行策略，需要调用`booster.execute_pipeline`来执行模型的训练，它会调用`scheduler`管理模型的前后向操作。
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
开始训练模型
```python
for epoch in range(NUM_EPOCH):
    train_epoch(epoch, model, optimizer, criterion, lr_scheduler, train_dataloader, booster, coordinator)
```
<!-- doc-test-command: echo  -->

# 使用 Colossal-AI （从数据并行到异构并行）加速 ViT 训练详解

作者：Yuxuan Lou

**示例代码**

- [Colossal-AI Examples ViT on Cifar10](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/image/vision_transformer)

**相关文献**
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)


## 引言

在这个ViT模型的样例中，Colossal-AI 提供了三种不同的并行技术来加速模型训练：数据并行，流水线并行和张量并行。我们将展示如何使用这三种并行技术在 CIFAR-10 数据集上训练 ViT。为了运行项目，需要2-4个 GPU。


## 目录
1. Colossal-AI 安装方法
2. 使用数据并行训练 ViT 步骤
3. 使用数据流水线并行训练 ViT 步骤
4. 使用张量并行或异构并行训练 ViT 步骤

## Colossal-AI 安装
可以通过 Python 的官方索引来安装 Colossal-AI 软件包。
```bash
pip install colossalai
```



## 数据并行
数据并行是实现加速模型训练的基本方法。通过两步可以实现训练的数据并行：
1. 构建一个配置文件
2. 在训练脚本中修改很少的几行代码

### 构建配置文件 (`data_parallel/config.py`)
为了使用 Colossal-AI，第一步是构建配置文件。并且，在这里有两种变量：

1. **Colossal-AI 功能配置**

Colossal-AI 提供了一系列的功能来加快训练速度（包括模型并行，混合精度，零冗余优化器等）。每个功能都是由配置文件中的相应字段定义的。如果我们只用到数据并行，那么我们只需要具体说明并行模式。在本例中，我们使用 PyTorch 最初提出的混合精度训练，只需要定义混合精度配置 `fp16 = dict(mode=AMP_TYPE.TORCH)` 。

2. **全局超参数**

全局超参数包括特定于模型的超参数、训练设置、数据集信息等。

```python
from colossalai.amp import AMP_TYPE
# ViT Base
BATCH_SIZE = 256
DROP_RATE = 0.1
NUM_EPOCHS = 300
# mix precision
fp16 = dict(
    mode=AMP_TYPE.TORCH,
)
gradient_accumulation = 16
clip_grad_norm = 1.0
dali = dict(
    gpu_aug=True,
    mixup_alpha=0.2
)
```

### 修改训练脚本 (`/data_parallel/train_with_cifar10.py`)

#### 导入模块
- Colossal-AI 相关模块
```python
import colossalai
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.lr_scheduler import LinearWarmupLR
from colossalai.nn.metric import Accuracy
from colossalai.trainer import Trainer, hooks
```

- 其他模块
```python
import os
import torch
from timm.models import vit_base_patch16_224
from torchvision import transforms
from torchvision.datasets import CIFAR10
```

#### 启动 Colossal-AI

在训练脚本中，在构建好配置文件后，需要为 Colossal-AI 初始化分布式环境。我们将此过程称为 `launch` 。在 Colossal-AI 中，我们提供了几种启动方法来初始化分布式后端。在大多数情况下，您可以使用 `colossalai.launch` 和 `colossalai.get_default_parser ` 来实现使用命令行传递参数。此外，Colossal-AI 可以利用 PyTorch 提供的现有启动工具，正如许多用户通过使用熟知的 `colossalai.launch_from_torch` 那样。更多详细信息，您可以查看相关[文档](https://www.colossalai.org/docs/basics/launch_colossalai)。


```python
# initialize distributed setting
parser = colossalai.get_default_parser()
args = parser.parse_args()
colossalai.launch_from_torch(config=args.config)
disable_existing_loggers()
logger = get_dist_logger()
```

初始化后，您可以使用 `colossalai.core.global_context` 访问配置文件中的变量。

```python
#access parameters
print(gpc.config.BATCH_SIZE)
```

#### 构建模型

如果只需要数据并行性，则无需对模型代码进行任何更改。这里，我们使用 `timm` 中的 `vit_base_patch16_224`。

```python
# build model
model = vit_base_patch16_224(drop_rate=0.1, num_classes=gpc.config.NUM_CLASSES)
```

#### 构建 CIFAR-10 数据加载器
`colossalai.utils.get_dataloader` 可以帮助您轻松构建数据加载器。

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
# build dataloader
train_dataloader, test_dataloader = build_cifar(gpc.config.BATCH_SIZE)
```

#### 定义优化器，损失函数和学习率调度器

Colossal-AI 提供了自己的优化器、损失函数和学习率调度器。PyTorch 的这些组件与Colossal-AI也兼容。

```python
# build optimizer
optimizer = colossalai.nn.Lamb(model.parameters(), lr=1.8e-2, weight_decay=0.1)
# build loss
criterion = torch.nn.CrossEntropyLoss()
# lr_scheduler
lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=50, total_steps=gpc.config.NUM_EPOCHS)
```

#### 启动用于训练的 Colossal-AI 引擎

Engine 本质上是对模型、优化器和损失函数的封装类。当我们使用 `colossalai.initialize` ，将返回一个 engine 对象，并且它已经按照配置文件中的指定内容，配置了梯度剪裁、梯度累积和零冗余优化器等功能。之后，基于 Colossal-AI 的 engine 我们可以进行模型训练。

```python
engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        model, optimizer, criterion, train_dataloader, test_dataloader
    )
```

#### 训练：Trainer 应用程序编程接口
Trainer 是一个更高级的封装类，用户可以用更少的代码就可以实现训练。通过传递 engine 对象很容易创建 trainer 对象。

此外，在 trainer 中，用户可以自定义一些挂钩，并将这些挂钩连接到 trainer 对象。钩子对象将根据训练方案定期执行生命周期方法。例如，`LRSchedulerHook` 将执行`lr_scheduler.step()` 在 `after_train_iter` 或 `after_train_epoch` 阶段更新模型的学习速率。

```python
# build trainer
trainer = Trainer(engine=engine, logger=logger)
# build hooks
hook_list = [
    hooks.LossHook(),
    hooks.AccuracyHook(accuracy_func=MixupAccuracy()),
    hooks.LogMetricByEpochHook(logger),
    hooks.LRSchedulerHook(lr_scheduler, by_epoch=True),
    # comment if you do not need to use the hooks below
    hooks.SaveCheckpointHook(interval=1, checkpoint_dir='./ckpt'),
    hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
]
```

使用 `trainer.fit` 进行训练:

```python
# start training
trainer.fit(
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    epochs=gpc.config.NUM_EPOCHS,
    hooks=hook_list,
    display_progress=True,
    test_interval=1
)
```

### 开始训练
`DATA` 是自动下载和存储 CIFAR-10 数据集的文件路径。

`<NUM_GPUs>` 是要用于使用 CIFAR-10 数据集，以数据并行方式训练 ViT 的 GPU 数。

```bash
export DATA=<path_to_data>
# If your torch >= 1.10.0
torchrun --standalone --nproc_per_node <NUM_GPUs>  train_dp.py --config ./configs/config_data_parallel.py
# If your torch >= 1.9.0
# python -m torch.distributed.run --standalone --nproc_per_node= <NUM_GPUs> train_dp.py --config ./configs/config_data_parallel.py
# Otherwise
# python -m torch.distributed.launch --nproc_per_node <NUM_GPUs> --master_addr <node_name> --master_port 29500 train_dp.py --config ./configs/config.py
```



## 流水线并行
除了数据并行性，Colossal-AI 还支持流水线并行。具体而言，Colossal-AI 使用 NVIDIA 引入的 1F1B 流水线。更多详细信息，您可以查看相关[文档](https://www.colossalai.org/tutorials/features/pipeline_parallel)。

### 构建配置文件(`hybrid_parallel/configs/vit_pipeline.py`)
要在数据并行的基础上应用流水线并行，只需添加一个 **parallel dict**
```python
from colossalai.amp import AMP_TYPE
parallel = dict(
    pipeline=2
)
# pipeline config
NUM_MICRO_BATCHES = parallel['pipeline']
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LENGTH, HIDDEN_SIZE)
fp16 = dict(mode=AMP_TYPE.NAIVE)
clip_grad_norm = 1.0
```

其他配置：
```python
# hyperparameters
# BATCH_SIZE is as per GPU
# global batch size = BATCH_SIZE x data parallel size
BATCH_SIZE = 256
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.3
NUM_EPOCHS = 300
WARMUP_EPOCHS = 32
# model config
IMG_SIZE = 224
PATCH_SIZE = 16
HIDDEN_SIZE = 768
DEPTH = 12
NUM_HEADS = 12
MLP_RATIO = 4
NUM_CLASSES = 10
CHECKPOINT = True
SEQ_LENGTH = (IMG_SIZE // PATCH_SIZE) ** 2 + 1  # add 1 for cls token
```

### 构建流水线模型 (`/hybrid_parallel/model/vit.py`)
Colossal-AI 提供了两种从现有模型构建流水线模型的方法。
- `colossalai.builder.build_pipeline_model_from_cfg`
- `colossalai.builder.build_pipeline_model`

此外，您还可以使用 Colossal-AI 从头开始构建流水线模型。
```python
import math
from typing import Callable
import inspect
import torch
from colossalai import nn as col_nn
from colossalai.registry import LAYERS, MODELS
from colossalai.logging import get_dist_logger
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
from colossalai.builder.pipeline import partition_uniform
from torch import dtype, nn
from model_zoo.vit.vit import ViTBlock, ViTEmbedding, ViTHead
@MODELS.register_module
class PipelineVisionTransformer(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 depth: int = 12,
                 num_heads: int = 12,
                 dim: int = 768,
                 mlp_ratio: int = 4,
                 attention_dropout: float = 0.,
                 dropout: float = 0.1,
                 drop_path: float = 0.,
                 layernorm_epsilon: float = 1e-6,
                 activation: Callable = nn.functional.gelu,
                 representation_size: int = None,
                 dtype: dtype = None,
                 bias: bool = True,
                 checkpoint: bool = False,
                 init_method: str = 'torch',
                 first_stage=True,
                 last_stage=True,
                 start_idx=None,
                 end_idx=None,):
        super().__init__()
        layers = []
        if first_stage:
            embed = ViTEmbedding(img_size=img_size,
                                 patch_size=patch_size,
                                 in_chans=in_chans,
                                 embedding_dim=dim,
                                 dropout=dropout,
                                 dtype=dtype,
                                 init_method=init_method)
            layers.append(embed)
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        if start_idx is None and end_idx is None:
            start_idx = 0
            end_idx = depth
        blocks = [
            ViTBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attention_dropout=attention_dropout,
                dropout=dropout,
                drop_path=dpr[i],
                activation=activation,
                dtype=dtype,
                bias=bias,
                checkpoint=checkpoint,
                init_method=init_method,
            ) for i in range(start_idx, end_idx)
        ]
        layers.extend(blocks)
        if last_stage:
            norm = col_nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
            head = ViTHead(dim=dim,
                           num_classes=num_classes,
                           representation_size=representation_size,
                           dtype=dtype,
                           bias=bias,
                           init_method=init_method)
            layers.extend([norm, head])
        self.layers = nn.Sequential(
            *layers
        )
    def forward(self, x):
        x = self.layers(x)
        return x
def _filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}
def _build_pipeline_vit(module_cls, num_layers, num_chunks, device=torch.device('cuda'), **kwargs):
    logger = get_dist_logger()
    if gpc.is_initialized(ParallelMode.PIPELINE):
        pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
        pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    else:
        pipeline_size = 1
        pipeline_rank = 0
    rank = gpc.get_global_rank()
    parts = partition_uniform(num_layers, pipeline_size, num_chunks)[pipeline_rank]
    models = []
    for start, end in parts:
        kwargs['first_stage'] = start == 0
        kwargs['last_stage'] = end == num_layers
        kwargs['start_idx'] = start
        kwargs['end_idx'] = end
        logger.info(f'Rank{rank} build layer {start}-{end}, {end-start}/{num_layers} layers')
        chunk = module_cls(**_filter_kwargs(module_cls.__init__, kwargs)).to(device)
        models.append(chunk)
    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)
    return model
def build_pipeline_vit(num_layers, num_chunks, device=torch.device('cuda'), **kwargs):
    return _build_pipeline_vit(PipelineVisionTransformer, num_layers, num_chunks, device, **kwargs)
```

### 修改训练脚本 (`/hybrid_parallel/train_with_cifar10.py`)

#### 导入模块
```python
from colossalai.engine.schedule import (InterleavedPipelineSchedule,
                                        PipelineSchedule)
from colossalai.utils import MultiTimer
import os
import colossalai
import torch
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn import CrossEntropyLoss
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.utils import is_using_pp, get_dataloader
from model.vit import build_pipeline_vit
from model_zoo.vit.vit import _create_vit_model
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import CIFAR10
```

#### 启动 Colossal-AI
`colossalai.utils.is_using_pp` 可以帮您检查配置文件是否满足流水线并行的要求。

```python
# initialize distributed setting
parser = colossalai.get_default_parser()
args = parser.parse_args()
# launch from torch
colossalai.launch_from_torch(config=args.config)
# get logger
logger = get_dist_logger()
logger.info("initialized distributed environment", ranks=[0])
if hasattr(gpc.config, 'LOG_PATH'):
    if gpc.get_global_rank() == 0:
        log_path = gpc.config.LOG_PATH
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        logger.log_to_file(log_path)
use_pipeline = is_using_pp()
```

#### 定义模型

```python
# create model
model_kwargs = dict(img_size=gpc.config.IMG_SIZE,
                    patch_size=gpc.config.PATCH_SIZE,
                    dim=gpc.config.HIDDEN_SIZE,
                    depth=gpc.config.DEPTH,
                    num_heads=gpc.config.NUM_HEADS,
                    mlp_ratio=gpc.config.MLP_RATIO,
                    num_classes=gpc.config.NUM_CLASSES,
                    init_method='jax',
                    checkpoint=gpc.config.CHECKPOINT)
if use_pipeline:
    model = build_pipeline_vit(num_layers=model_kwargs['depth'], num_chunks=1, **model_kwargs)
else:
    model = _create_vit_model(**model_kwargs)
```

#### 计算参数个数

您可以轻松计算不同流水线阶段上的模型参数个数。

```
# count number of parameters
total_numel = 0
for p in model.parameters():
    total_numel += p.numel()
if not gpc.is_initialized(ParallelMode.PIPELINE):
    pipeline_stage = 0
else:
    pipeline_stage = gpc.get_local_rank(ParallelMode.PIPELINE)
logger.info(f"number of parameters: {total_numel} on pipeline stage {pipeline_stage}")
```

#### 构建数据加载器，优化器等组件

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


# create dataloaders
train_dataloader , test_dataloader = build_cifar()
# create loss function
criterion = CrossEntropyLoss(label_smoothing=0.1)
# create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=gpc.config.LEARNING_RATE, weight_decay=gpc.config.WEIGHT_DECAY)
# create lr scheduler
lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer,
                                       total_steps=gpc.config.NUM_EPOCHS,
                                       warmup_steps=gpc.config.WARMUP_EPOCHS)
```

#### 启动 Colossal-AI 引擎

```python
# initialize
engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model=model,
                                                                     optimizer=optimizer,
                                                                     criterion=criterion,
                                                                     train_dataloader=train_dataloader,
                                                                     test_dataloader=test_dataloader)
logger.info("Engine is built", ranks=[0])
```

#### 训练：基于engine

在数据并行示例中，我们展示了如何使用 Trainer API 训练模型。我们还可以直接训练基于 engine 的模型。通过这种方式，您可以使用更多功能自定义训练方法。

```python
data_iter = iter(train_dataloader)
for epoch in range(gpc.config.NUM_EPOCHS):
    # training
    engine.train()
    if gpc.get_global_rank() == 0:
        description = 'Epoch {} / {}'.format(
            epoch,
            gpc.config.NUM_EPOCHS
        )
        progress = tqdm(range(len(train_dataloader)), desc=description)
    else:
        progress = range(len(train_dataloader))
    for _ in progress:
        engine.zero_grad()
        engine.execute_schedule(data_iter, return_output_label=False)
        engine.step()
        lr_scheduler.step()
```

### 开始训练
```bash
export DATA=<path_to_dataset>
# If your torch >= 1.10.0
torchrun --standalone --nproc_per_node <NUM_GPUs>  train_hybrid.py --config ./configs/config_pipeline_parallel.py
# If your torch >= 1.9.0
# python -m torch.distributed.run --standalone --nproc_per_node= <NUM_GPUs> train_hybrid.py --config ./configs/config_pipeline_parallel.py
```




## 张量并行和异构并行
张量并行将每个权重参数跨多个设备进行分区，以减少内存负载。Colossal-AI 支持 1D、2D、2.5D 和 3D 张量并行。此外，还可以将张量并行、流水线并行和数据并行结合起来，实现混合并行。Colossal-AI 还提供了一种简单的方法来应用张量并行和混合并行。只需在配置文件中更改几行代码即可实现流水线并行。

### 构造您的配置文件 (`/hybrid_parallel/configs/vit_1d_tp2_pp2.py`)
使用张量并行，只需将相关信息添加到 **parallel dict**。具体而言，`TENSOR_PARALLEL_MODE` 可以是“1d”、“2d”、“2.5d”、“3d”。不同并行度的大小应满足：`#GPUs = pipeline parallel size x tensor parallel size x data parallel size`。在指定 GPU 数量、流水线并行大小和张量并行大小后 `data parallel size` 会自动计算。

```python
from colossalai.amp import AMP_TYPE
# parallel setting
TENSOR_PARALLEL_SIZE = 2
TENSOR_PARALLEL_MODE = '1d'
parallel = dict(
    pipeline=2,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE)
)
fp16 = dict(mode=AMP_TYPE.NAIVE)
clip_grad_norm = 1.0
# pipeline config
NUM_MICRO_BATCHES = parallel['pipeline']
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LENGTH, HIDDEN_SIZE)
```

其他配置:
```python
# hyperparameters
# BATCH_SIZE is as per GPU
# global batch size = BATCH_SIZE x data parallel size
BATCH_SIZE = 256
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.3
NUM_EPOCHS = 300
WARMUP_EPOCHS = 32
# model config
IMG_SIZE = 224
PATCH_SIZE = 16
HIDDEN_SIZE = 768
DEPTH = 12
NUM_HEADS = 12
MLP_RATIO = 4
NUM_CLASSES = 10
CHECKPOINT = True
SEQ_LENGTH = (IMG_SIZE // PATCH_SIZE) ** 2 + 1  # add 1 for cls token
```

### 开始训练
```bash
export DATA=<path_to_dataset>
# If your torch >= 1.10.0
torchrun --standalone --nproc_per_node <NUM_GPUs>  train_hybrid.py --config ./configs/config_hybrid_parallel.py
# If your torch >= 1.9.0
# python -m torch.distributed.run --standalone --nproc_per_node= <NUM_GPUs> train_hybrid.py --config ./configs/config_hybrid_parallel.py
```

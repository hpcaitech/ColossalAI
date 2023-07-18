# Step By Step: Accelerate ViT Training With Colossal-AI (From Data Parallel to Hybrid Parallel)

Author: Yuxuan Lou

**Example Code**

- [Colossal-AI Examples ViT on Cifar10](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/image/vision_transformer)

**Related Paper**
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)


## Introduction

In this example for ViT model, Colossal-AI provides three different parallelism techniques which accelerate model training: data parallelism, pipeline parallelism and tensor parallelism.
We will show you how to train ViT on CIFAR-10 dataset with these parallelism techniques. To run this example, you will need 2-4 GPUs.


## Table of Contents
1. Colossal-AI installation
2. Steps to train ViT with data parallelism
3. Steps to train ViT with pipeline parallelism
4. Steps to train ViT with tensor parallelism or hybrid parallelism

## Colossal-AI Installation
You can install Colossal-AI package and its dependencies with PyPI.
```bash
pip install colossalai
```



## Data Parallelism
Data parallelism is one basic way to accelerate model training process. You can apply data parallelism to training by only two steps:
1. Define a configuration file
2. Change a few lines of code in train script

### Define your configuration file (`data_parallel/config.py`)
To use Colossal-AI, the first step is to define a configuration file. And there are two kinds of variables here:

1. **Colossal-AI feature specification**

There is an array of features Colossal-AI provides to speed up training (parallel mode, mixed precision, ZeRO, etc.). Each feature is defined by a corresponding field in the config file. If we apply data parallel only, we do not need to specify the parallel mode. In this example, we use mixed precision training natively provided by PyTorch by define the mixed precision configuration `fp16 = dict(mode=AMP_TYPE.TORCH)`.

2. **Global hyper-parameters**

Global hyper-parameters include model-specific hyper-parameters, training settings, dataset information, etc.

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

### Modify train script (`/data_parallel/train_with_cifar10.py`)

#### Import modules
- Colossal-AI related modules
```python
import colossalai
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.lr_scheduler import LinearWarmupLR
from colossalai.nn.metric import Accuracy
from colossalai.trainer import Trainer, hooks
```

- Other modules
```python
import os

import torch
from timm.models import vit_base_patch16_224


from torchvision import transforms
from torchvision.datasets import CIFAR10
```

#### Launch Colossal-AI

In train script,  you need to initialize the distributed environment for Colossal-AI after your config file is prepared. We call this process `launch`. In Colossal-AI, we provided several launch methods to initialize the distributed backend. In most cases, you can use `colossalai.launch` and `colossalai.get_default_parser` to pass the parameters via command line. Besides, Colossal-AI can utilize the existing launch tool provided by PyTorch as many users are familiar with by using `colossalai.launch_from_torch`. For more details, you can view the related [documents](https://www.colossalai.org/docs/basics/launch_colossalai).

```python
# initialize distributed setting
parser = colossalai.get_default_parser()
args = parser.parse_args()
colossalai.launch_from_torch(config=args.config)

disable_existing_loggers()
logger = get_dist_logger()
```

After initialization, you can access the variables in the config file by using `colossalai.core.global_context`.

```python
#access parameters
print(gpc.config.BATCH_SIZE)
```

#### Build Model

If only data parallelism is required, you do not need to make any changes to your model. Here, we use `vit_base_patch16_224` from `timm`.
```python
# build model
model = vit_base_patch16_224(drop_rate=0.1, num_classes=gpc.config.NUM_CLASSES)
```

#### Build CIFAR-10 Dataloader
`colossalai.utils.get_dataloader` can help you build dataloader easily.

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

#### Define optimizer, loss function and LR scheduler

Colossal-AI provides its own optimizer, loss function and LR scheduler. Those from PyTorch are also compatible.

```python
# build optimizer
optimizer = colossalai.nn.Lamb(model.parameters(), lr=1.8e-2, weight_decay=0.1)

# build loss
criterion = torch.nn.CrossEntropyLoss()

# lr_scheduler
lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=50, total_steps=gpc.config.NUM_EPOCHS)
```

#### Start Colossal-AI engine

Engine is essentially a wrapper class for model, optimizer and loss function. When we call `colossalai.initialize`, an engine object will be returned, and it has already been equipped with functionalities such as gradient clipping, gradient accumulation and zero optimizer as specified in your configuration file. Further model training is based on Colossal-AI engine.

```python
engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        model, optimizer, criterion, train_dataloader, test_dataloader
    )
```

#### Train: Trainer API
Trainer is a more high-level wrapper for the user to execute training with fewer lines of code. It is easy to create a trainer object by passing the engine object.

Besides, In trainer, the user can customize some hooks and attach these hooks to the trainer object. A hook object will execute life-cycle methods periodically based on the training scheme. For example, The `LRSchedulerHook` will execute `lr_scheduler.step()` to update the learning rate of the model during either `after_train_iter` or `after_train_epoch` stages.

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

Use `trainer.fit` for training:

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

### Start training
`DATA` is the filepath where CIFAR-10 dataset will be automatically downloaded and stored.

`<NUM_GPUs>` is the number of GPUs you want to use to train ViT on CIFAR-10 with data parallelism.

```bash
export DATA=<path_to_data>
# If your torch >= 1.10.0
torchrun --standalone --nproc_per_node <NUM_GPUs>  train_dp.py --config ./configs/config_data_parallel.py
# If your torch >= 1.9.0
# python -m torch.distributed.run --standalone --nproc_per_node= <NUM_GPUs> train_dp.py --config ./configs/config_data_parallel.py
# Otherwise
# python -m torch.distributed.launch --nproc_per_node <NUM_GPUs> --master_addr <node_name> --master_port 29500 train_dp.py --config ./configs/config.py
```



## Pipeline Parallelism
Aside from data parallelism, Colossal-AI also support pipeline parallelism. In specific, Colossal-AI uses 1F1B pipeline introduced by NVIDIA. For more details, you can view the related [documents](https://www.colossalai.org/tutorials/features/pipeline_parallel).

### Define your configuration file(`hybrid_parallel/configs/vit_pipeline.py`)
To apply pipeline parallel on the data parallel basis, you only need to add a **parallel dict**
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

Other configs：
```python
# hyper parameters
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

### Build pipeline model (`/hybrid_parallel/model/vit.py`)
Colossal-AI provides two methods to build a pipeline model from the existing model.
- `colossalai.builder.build_pipeline_model_from_cfg`
- `colossalai.builder.build_pipeline_model`

Besides, you can also build a pipeline model from scratch with Colossal-AI.
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

### Modify train script (`/hybrid_parallel/train_with_cifar10.py`)

#### Import modules
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

#### Launch Colossal-AI
`colossalai.utils.is_using_pp` can help check whether pipeline parallelism is required in config file.

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

#### Define model

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

#### Count number of parameters

You can count model parameters on different pipeline stages easily.

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

#### Build dataloader, optimizer, etc.

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

#### Start Colossal-AI engine

```python
# initialize
engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model=model,
                                                                     optimizer=optimizer,
                                                                     criterion=criterion,
                                                                     train_dataloader=train_dataloader,
                                                                     test_dataloader=test_dataloader)

logger.info("Engine is built", ranks=[0])
```

#### Train: based on engine

In the data parallelism example, we show how to train a model with Trainer API. We can also directly train a model based on engine. In this way, you can customize your training with more features.

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

### Start training
```bash
export DATA=<path_to_dataset>
# If your torch >= 1.10.0
torchrun --standalone --nproc_per_node <NUM_GPUs>  train_hybrid.py --config ./configs/config_pipeline_parallel.py
# If your torch >= 1.9.0
# python -m torch.distributed.run --standalone --nproc_per_node= <NUM_GPUs> train_hybrid.py --config ./configs/config_pipeline_parallel.py
```




## Tensor Parallelism and Hybrid Parallelism
Tensor parallelism partitions each weight parameter across multiple devices in order to reduce memory load. Colossal-AI support 1D, 2D, 2.5D and 3D tensor parallelism. Besides, you can combine tensor parallelism with pipeline parallelism and data parallelism to reach hybrid parallelism. Colossal-AI also provides an easy way to apply tensor parallelism and hybrid parallelism. On the basis of pipeline parallelism, a few lines of code changing in config file is all you need.

### Define your configuration file(`/hybrid_parallel/configs/vit_1d_tp2_pp2.py`)
To use tensor parallelism, you only need to add related information to the **parallel dict**. To be specific, `TENSOR_PARALLEL_MODE` can be '1d', '2d', '2.5d', '3d'. And the size of different parallelism should satisfy: `#GPUs = pipeline parallel size x tensor parallel size x data parallel size`.  `data parallel size` will automatically computed after you specify the number of GPUs, pipeline parallel size and tensor parallel size.

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

Other configs:
```python
# hyper parameters
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

### Start training
```bash
export DATA=<path_to_dataset>
# If your torch >= 1.10.0
torchrun --standalone --nproc_per_node <NUM_GPUs>  train_hybrid.py --config ./configs/config_hybrid_parallel.py
# If your torch >= 1.9.0
# python -m torch.distributed.run --standalone --nproc_per_node= <NUM_GPUs> train_hybrid.py --config ./configs/config_hybrid_parallel.py
```

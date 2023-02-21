# 使用流水并行训练 ViT

作者: Hongxin Liu, Yongbin Li

**示例代码**
- [ColossalAI-Examples Pipeline Parallel ViT](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/image/vision_transformer/pipeline_parallel)

**相关论文**
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

## 引言

在本教程中，你将学习如何使用流水并行从头开始训练用于图像分类的 Vision Transformer (ViT)。流水并行是一种模型并行，主要针对 GPU 内存不能满足模型容量的情况。
通过使用流水并行，我们将原始模型分割成多个阶段，每个阶段保留原始模型的一部分。我们假设你的 GPU 内存不能容纳 ViT/L-16，而你的内存可以容纳这个模型。

##  目录

在本教程中，我们将介绍:

1. 基于 [TIMM](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py) 定义 ViT 模型
2. 处理数据集
3. 使用流水并行训练 ViT

## 导入依赖库

```python
import os
from collections import OrderedDict
from functools import partial

import colossalai
import colossalai.nn as col_nn
import torch
import torch.nn as nn
from colossalai.builder import build_pipeline_model
from colossalai.engine.schedule import (InterleavedPipelineSchedule,
                                        PipelineSchedule)
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_dataloader
from timm.models import vision_transformer as vit
from torchvision import transforms
from torchvision.datasets import CIFAR10
```


## 定义 Vision Transformer 模型

总的来说, 我们提供3种方法来建立一个流水并行的模型:

1. `colossalai.builder.build_pipeline_model_from_cfg`
2. `colossalai.builder.build_pipeline_model`
3. 自己按阶段拆分模型

当你的内存能够容纳模型时，你可以使用前两种方法来建立你的模型，否则你必须自己分割模型。前两种方法首先在 CPU 上建立整个模型，然后分割模型，最后你可以直接把模型的相应部分移到 GPU 上。

`colossalai.builder.build_pipeline_model_from_cfg()` 接收一个模型的配置文件，它可以均匀地（按层）或平衡地（按参数大小）分割模型。

如果你熟悉 `PyTorch`, 你可以使用 `colossalai.builder.build_pipeline_model()` 它接收一个 `torch.nn.Sequential` 模型并按层均匀分割。

在本教程中，我们将修改 [TIMM/ViT](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py) to `torch.nn.Sequential`，然后使用 `colossalai.builder.build_pipeline_model()` 来建立流水线模型。

当数据是 **一个** `Tensor`, 你可以使用你的模型 `forward()` 中的位置参数来获得数据张量。对于流水线的第一阶段，`forward()` 的第一个位置参数是从数据加载器加载的数据张量。对于其他阶段，`forward()` 的第一个位置参数是上一阶段的输出张量。注意，如果该阶段不是最后一个阶段，则 `forward()` 的返回必须是一个 `Tensor`。

当数据是一个 `Tensor` 的 `dict`, 你可以使用你模型 `forward()` 的命名关键字参数来获得数据的 `dict`。

```python
class ViTEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, embed_layer=vit.PatchEmbed, drop_rate=0., distilled=False):
        super().__init__()
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.init_weights()

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return x

    def init_weights(self):
        vit.trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            vit.trunc_normal_(self.dist_token, std=.02)
        vit.trunc_normal_(self.cls_token, std=.02)
        self.apply(vit._init_vit_weights)


class ViTHead(nn.Module):
    def __init__(self, embed_dim=768, num_classes=1000, norm_layer=None, distilled=False, representation_size=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        self.num_classes = num_classes
        self.distilled = distilled
        self.num_features = embed_dim
        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.init_weights()

    def forward(self, x):
        x = self.norm(x)
        if self.distilled:
            x, x_dist = self.head(x[:, 0]), self.head_dist(x[:, 1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.pre_logits(x[:, 0])
            x = self.head(x)
        return x

    def init_weights(self):
        self.apply(vit._init_vit_weights)


def sequential_vit(img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                   num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                   drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=vit.PatchEmbed, norm_layer=None,
                   act_layer=None):
    norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
    act_layer = act_layer or nn.GELU
    embedding = ViTEmbedding(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                             embed_dim=embed_dim, embed_layer=embed_layer, drop_rate=drop_rate, distilled=distilled)
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
    blocks = [vit.Block(
        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
        for i in range(depth)]
    for block in blocks:
        block.apply(vit._init_vit_weights)
    head = ViTHead(embed_dim=embed_dim, num_classes=num_classes, norm_layer=norm_layer,
                   distilled=distilled, representation_size=representation_size)
    return nn.Sequential(embedding, *blocks, head)


def vit_large_patch16_224(**kwargs):
    model_kwargs = dict(embed_dim=1024, depth=24, num_heads=16, **kwargs)
    return sequential_vit(**model_kwargs)
```

## 处理数据集

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

## 使用流水并行训练 ViT

你可以在配置文件中设置流水并行的大小。`NUM_CHUNKS` 在使用交错流水线时很有用 (更多细节见 [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) )。
原始 batch 将会被分割为 `num_microbatches`, 每个阶段每次将加载一个 micro batch。如果你确定性地知道每个阶段输出张量的形状，你可以在配置文件中设置 `tensor_shape` 来减少通信。
我们的仓库会自动为用户生成合适的schedule来支持流水并行训练。如果你不需要模型的输出和标签，你可以在调用 `trainer.fit()` 时，将 `return_output_label` 设置为 `False`，这样能进一步减少 GPU 显存使用。

你应当使用 `export DATA=/path/to/cifar`。

```python
BATCH_SIZE = 16
NUM_EPOCHS = 60
NUM_CHUNKS = 1
CONFIG = dict(NUM_MICRO_BATCHES=4, parallel=dict(pipeline=2))


def train():
    disable_existing_loggers()
    parser = colossalai.get_default_parser()
    args = parser.parse_args()
    colossalai.launch_from_torch(backend=args.backend, config=CONFIG)
    logger = get_dist_logger()

    # build model
    model = vit_large_patch16_224()
    model = build_pipeline_model(model, num_chunks=NUM_CHUNKS, verbose=True)

    # build criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    # build dataloader
    train_dataloader, test_dataloader = build_cifar(BATCH_SIZE)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model, optimizer, criterion,
                                                                         train_dataloader, test_dataloader)
    timer = MultiTimer()

    trainer = Trainer(engine=engine, timer=timer, logger=logger)

    hook_list = [
        hooks.LossHook(),
        hooks.AccuracyHook(col_nn.metric.Accuracy()),
        hooks.LogMetricByEpochHook(logger),
    ]

    trainer.fit(train_dataloader=train_dataloader,
                epochs=NUM_EPOCHS,
                test_dataloader=test_dataloader,
                test_interval=1,
                hooks=hook_list,
                display_progress=True)
```

#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from pathlib import Path

# from colossalai.context import ParallelMode
from colossalai.engine import AMP_TYPE
from torchvision.transforms import AutoAugmentPolicy

IMG_SIZE = 32
PATCH_SIZE = 4
EMBED_SIZE = 256
HIDDEN_SIZE = 256
NUM_HEADS = 4
NUM_CLASSES = 10
NUM_BLOCKS = 7
DROP_RATE = 0.1

BATCH_SIZE = 512
LEARNING_RATE = 0.001
WEIGHT_DECAY = 3e-2

DATASET_PATH = Path(os.environ['DATA'])

model = dict(
    type='VisionTransformerFromConfig',
    embedding_cfg=dict(
        type='ViTPatchEmbedding3D',
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=3,
        embed_size=EMBED_SIZE,
        drop_prob=DROP_RATE,
    ),
    block_cfg=dict(
        type='ViTBlock',
        norm_cfg=dict(
            type='LayerNorm3D',
            normalized_shape=HIDDEN_SIZE,
            eps=1e-6,
            # input_parallel_mode=ParallelMode.PARALLEL_3D_INPUT,
            # weight_parallel_mode=ParallelMode.PARALLEL_3D_WEIGHT,
        ),
        attention_cfg=dict(
            type='ViTSelfAttention3D',
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=NUM_HEADS,
            attention_probs_dropout_prob=0.,
            hidden_dropout_prob=DROP_RATE,
        ),
        droppath_cfg=dict(type='VanillaViTDropPath', ),
        mlp_cfg=dict(
            type='ViTMLP3D',
            hidden_size=HIDDEN_SIZE,
            mlp_ratio=2,
            hidden_dropout_prob=DROP_RATE,
            hidden_act='gelu',
        ),
    ),
    norm_cfg=dict(type='LayerNorm3D',
                  normalized_shape=HIDDEN_SIZE,
                  eps=1e-6,
                #   input_parallel_mode=ParallelMode.PARALLEL_3D_INPUT,
                #   weight_parallel_mode=ParallelMode.PARALLEL_3D_WEIGHT,
                  ),
    head_cfg=dict(
        type='ViTHead3D',
        in_features=HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
    ),
    embed_dim=HIDDEN_SIZE,
    depth=NUM_BLOCKS,
    drop_path_rate=0.,
)

loss = dict(type='CrossEntropyLoss3D',
            # input_parallel_mode=ParallelMode.PARALLEL_3D_OUTPUT,
            # weight_parallel_mode=ParallelMode.PARALLEL_3D_WEIGHT,
            # reduction=True,
            )
# loss = dict(type='CrossEntropyLoss', label_smoothing=0.1)

optimizer = dict(type='AdamW', lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

train_data = dict(
    dataset=dict(
        type='CIFAR10Dataset',
        root=DATASET_PATH,
        transform_pipeline=[
            dict(type='RandomCrop', size=IMG_SIZE, padding=4),
            #    dict(type='RandomHorizontalFlip'),
            dict(type='AutoAugment', policy=AutoAugmentPolicy.CIFAR10),
            dict(type='ToTensor'),
            dict(type='Normalize',
                 mean=[0.4914, 0.4822, 0.4465],
                 std=[0.2023, 0.1994, 0.2010]),
        ]),
    dataloader=dict(batch_size=BATCH_SIZE,
                    pin_memory=True,
                    shuffle=True,
                    num_workers=1))

test_data = dict(dataset=dict(type='CIFAR10Dataset',
                              root=DATASET_PATH,
                              train=False,
                              transform_pipeline=[
                                  dict(type='Resize', size=IMG_SIZE),
                                  dict(type='ToTensor'),
                                  dict(type='Normalize',
                                       mean=[0.4914, 0.4822, 0.4465],
                                       std=[0.2023, 0.1994, 0.2010]),
                              ]),
                 dataloader=dict(batch_size=1000,
                                 pin_memory=True))

parallel = dict(
    data=1,
    pipeline=1,
    tensor=dict(mode='3d', size=8),
)

clip_grad = 1.0

engine = dict(
    schedule=None,
    gradient_handlers=None,
    gradient_accumulation=1,
    gradient_clipping=clip_grad,
)

num_epochs = 200

hooks = [
    dict(type='LogMetricByEpochHook'),
    dict(type='LogMemoryByEpochHook'),
    dict(
        type='Accuracy3DHook',
        # input_parallel_mode=ParallelMode.PARALLEL_3D_OUTPUT,
        # weight_parallel_mode=ParallelMode.PARALLEL_3D_WEIGHT,
    ),
    dict(type='LossHook'),
    dict(type='LRSchedulerHook',
         by_epoch=False,
         lr_scheduler_cfg=dict(type='CosineAnnealingWarmupLR',
                               warmup_epochs=10,
                               eta_min=1e-5)),
]

# fp16 = dict(mode=AMP_TYPE.TORCH, init_scale=2**6)

logging = dict(
    root_path=
    f"./vit_3d_cifar10_bs{BATCH_SIZE}_lr{LEARNING_RATE}_clip_grad{clip_grad}"
)

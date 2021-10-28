#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
from pathlib import Path

BATCH_SIZE = 128
IMG_SIZE = 224
NUM_CLS = 1000

# resnet 18
model = dict(
    type='VanillaResNet',
    block_type='ResNetBottleneck',
    layers=[3, 4, 6, 3],
    num_cls=NUM_CLS
)

train_data = dict(
    dataset=dict(
        type='CIFAR10Dataset',
        root=Path(os.environ['DATA']),
        transform_pipeline=[
            dict(type='RandomResizedCrop', size=IMG_SIZE),
            dict(type='RandomHorizontalFlip'),
            dict(type='ToTensor'),
            dict(type='Normalize', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    ),
    dataloader=dict(
        batch_size=64,
        pin_memory=True,
        num_workers=4,
        sampler=dict(
            type='DataParallelSampler',
            shuffle=True,
        )
    )
)

test_data = dict(
    dataset=dict(
        type='CIFAR10Dataset',
        root=Path(os.environ['DATA']),
        train=False,
        transform_pipeline=[
            dict(type='Resize', size=(IMG_SIZE, IMG_SIZE)),
            dict(type='ToTensor'),
            dict(type='Normalize', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    ),
    dataloader=dict(
        batch_size=BATCH_SIZE,
        pin_memory=True,
        num_workers=4,
    )
)

dist_initializer = [
    dict(type='DataParallelInitializer'),
]

parallelization = dict(
    pipeline=1,
    tensor=1,
    sequence=-1
)

optimizer = dict(
    type='Adam',
    lr=0.01
)

loss = dict(
    type='CrossEntropyLoss'
)

trainer = dict(
    max_epochs=5,
    max_iters=1000
)

amp = dict(
    fp16=None,
)

level = 2

parallel = dict(
    pipeline=dict(size=1),
    tensor=dict(size=1, mode=None)
)

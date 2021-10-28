#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os

IMG_SIZE = 224
BATCH_SIZE = 256

model = dict(
    type='VanillaResNet',
    block_type='ResNetBottleneck',
    layers=[3, 4, 6, 3],
    num_cls=10
)

train_data = dict(
    dataset=dict(
        type='CIFAR10Dataset',
        root=os.environ['DATA'],
        transform_pipeline=[
            dict(type='Resize', size=IMG_SIZE),
            dict(type='RandomCrop', size=IMG_SIZE, padding=4),
            dict(type='RandomHorizontalFlip'),
            dict(type='ToTensor'),
            dict(type='Normalize',
                 mean=[0.4914, 0.4822, 0.4465],
                 std=[0.2023, 0.1994, 0.2010]),
        ]
    ),
    dataloader=dict(
        batch_size=BATCH_SIZE,
        pin_memory=True,
        shuffle=True,
    )
)

test_data = dict(
    dataset=dict(
        type='CIFAR10Dataset',
        root=os.environ['DATA'],
        train=False,
        transform_pipeline=[
            dict(type='Resize', size=IMG_SIZE),
            dict(type='ToTensor'),
            dict(type='Normalize',
                 mean=[0.4914, 0.4822, 0.4465],
                 std=[0.2023, 0.1994, 0.2010]
                 ),
        ]
    ),
    dataloader=dict(
        batch_size=BATCH_SIZE,
        pin_memory=True,
    )
)

parallelization = dict(
    pipeline=1,
    tensor=dict(size=1, mode=None),
)

optimizer = dict(
    type='Adam',
    lr=0.01
)

loss = dict(
    type='CrossEntropyLoss'
)

max_epochs = 100

from colossalai.engine import AMP_TYPE

fp16 = dict(
    mode=AMP_TYPE.APEX,
    opt_level='O2',
)

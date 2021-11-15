#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from pathlib import Path

from colossalai.context import ParallelMode
from colossalai.engine import AMP_TYPE

try:
    import model_zoo
except:
    print('You need to set model_zoo to your PYTHONPATH to use the models in the collection')

BATCH_SIZE = 512
IMG_SIZE = 32
NUM_EPOCHS = 60

train_data = dict(
    dataset=dict(
        type='CIFAR10Dataset',
        root=Path(os.environ['DATA']),
        transform_pipeline=[
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
        num_workers=2,
        shuffle=True,
    )
)

test_data = dict(
    dataset=dict(
        type='CIFAR10Dataset',
        root=Path(os.environ['DATA']),
        train=False,
        transform_pipeline=[
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
        num_workers=2,
    )
)

optimizer = dict(
    type='Adam',
    lr=0.001
)

loss = dict(
    type='CrossEntropyLoss3D',
    input_parallel_mode=ParallelMode.PARALLEL_3D_OUTPUT,
    weight_parallel_mode=ParallelMode.PARALLEL_3D_WEIGHT,
)

model = dict(
    type='vit_tiny_3d_patch4_32',
    drop_rate=0.1,
)

hooks = [
    dict(type='LogMetricByEpochHook'),
    dict(type='LogTimingByEpochHook'),
    dict(type='LogMemoryByEpochHook'),
    dict(
        type='Accuracy3DHook',
        input_parallel_mode=ParallelMode.PARALLEL_3D_OUTPUT,
        weight_parallel_mode=ParallelMode.PARALLEL_3D_WEIGHT,
    ),
    dict(type='LossHook'),
    dict(type='TensorboardHook', log_dir='./tfb_logs'),
    dict(
        type='LRSchedulerHook',
        by_epoch=True,
        lr_scheduler_cfg=dict(
            type='LinearWarmupLR',
            warmup_steps=5
        )
    ),
    # dict(type='SaveCheckpointHook', interval=5, checkpoint_dir='./ckpt'),
    # dict(type='LoadCheckpointHook', epoch=20, checkpoint_dir='./ckpt')
]

parallel = dict(
    pipeline=dict(size=1),
    tensor=dict(size=8, mode='3d'),
)

fp16 = dict(
    mode=AMP_TYPE.PARALLEL,
    initial_scale=2 ** 8
)

logging = dict(
    root_path='./logs'
)

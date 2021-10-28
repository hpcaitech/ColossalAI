import os
from pathlib import Path

BATCH_SIZE = 128
IMG_SIZE = 32

# resnet 50
model = dict(
    type='VanillaResNet',
    block_type='ResNetBottleneck',
    layers=[3, 4, 6, 3],
    num_cls=10
)

train_data = dict(
    dataset=dict(
        type='CIFAR10Dataset',
        root=Path(os.environ['DATA']),
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
        num_workers=4,
        shuffle=True
    )
)

test_data = dict(
    dataset=dict(
        type='CIFAR10Dataset',
        root=Path(os.environ['DATA']),
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
        num_workers=4,
        shuffle=True
    )
)

optimizer = dict(
    type='SGD',
    lr=0.2,
    momentum=0.9,
    weight_decay=5e-4
)

loss = dict(
    type='CrossEntropyLoss',
)

parallel = dict(
    pipeline=dict(size=1),
    tensor=dict(size=1, mode=None),
)

hooks = [
    dict(type='LogMetricByEpochHook'),
    dict(type='AccuracyHook'),
    dict(type='LossHook'),
    dict(type='TensorboardHook', log_dir='./tfb_logs'),
    dict(type='SaveCheckpointHook', interval=5, checkpoint_dir='./ckpt'),
    # dict(type='LoadCheckpointHook', epoch=20, checkpoint_dir='./ckpt')
]

# fp16 = dict(
#     mode=AMP_TYPE.PARALLEL,
#     initial_scale=1
# )

lr_scheduler = dict(
    type='CosineAnnealingLR',
    T_max=200
)

num_epochs = 200

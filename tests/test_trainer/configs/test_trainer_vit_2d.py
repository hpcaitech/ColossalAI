import os
from pathlib import Path

from colossalai.engine import AMP_TYPE

BATCH_SIZE = 512
IMG_SIZE = 32
PATCH_SIZE = 4
DIM = 512
NUM_ATTENTION_HEADS = 8
SUMMA_DIM = 2
NUM_CLASSES = 10
DEPTH = 6
num_epochs = 60

train_data = dict(
    dataset=dict(type='CIFAR10Dataset',
                 root=Path(os.environ['DATA']),
                 transform_pipeline=[
                     dict(type='Resize', size=IMG_SIZE),
                     dict(type='RandomCrop', size=IMG_SIZE, padding=4),
                     dict(type='RandomHorizontalFlip'),
                     dict(type='ToTensor'),
                     dict(type='Normalize',
                          mean=[0.4914, 0.4822, 0.4465],
                          std=[0.2023, 0.1994, 0.2010]),
                 ]),
    dataloader=dict(
        batch_size=BATCH_SIZE,
        pin_memory=True,
        # num_workers=1,
        shuffle=True,
    ))

test_data = dict(
    dataset=dict(type='CIFAR10Dataset',
                 root=Path(os.environ['DATA']),
                 train=False,
                 transform_pipeline=[
                     dict(type='Resize', size=IMG_SIZE),
                     dict(type='ToTensor'),
                     dict(type='Normalize',
                          mean=[0.4914, 0.4822, 0.4465],
                          std=[0.2023, 0.1994, 0.2010]),
                 ]),
    dataloader=dict(
        batch_size=400,
        pin_memory=True,
        # num_workers=1,
    ))

optimizer = dict(type='Adam', lr=0.001, weight_decay=0)

loss = dict(type='CrossEntropyLoss2D', )

model = dict(
    type='VisionTransformerFromConfig',
    tensor_splitting_cfg=dict(type='ViTInputSplitter2D', ),
    embedding_cfg=dict(
        type='ViTPatchEmbedding2D',
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=DIM,
    ),
    token_fusion_cfg=dict(type='ViTTokenFuser2D',
                          img_size=IMG_SIZE,
                          patch_size=PATCH_SIZE,
                          embed_dim=DIM,
                          drop_rate=0.1),
    norm_cfg=dict(
        type='LayerNorm2D',
        normalized_shape=DIM,
        eps=1e-6,
    ),
    block_cfg=dict(
        type='ViTBlock',
        attention_cfg=dict(
            type='ViTSelfAttention2D',
            hidden_size=DIM,
            num_attention_heads=NUM_ATTENTION_HEADS,
            attention_dropout_prob=0.,
            hidden_dropout_prob=0.1,
        ),
        droppath_cfg=dict(type='VanillaViTDropPath', ),
        mlp_cfg=dict(type='ViTMLP2D',
                     in_features=DIM,
                     dropout_prob=0.1,
                     mlp_ratio=1),
        norm_cfg=dict(
            type='LayerNorm2D',
            normalized_shape=DIM,
            eps=1e-6,
        ),
    ),
    head_cfg=dict(
        type='ViTHead2D',
        hidden_size=DIM,
        num_classes=NUM_CLASSES,
    ),
    embed_dim=DIM,
    depth=DEPTH,
    drop_path_rate=0.,
)

hooks = [
    dict(type='LogMetricByEpochHook'),
    dict(type='LogTimingByEpochHook'),
    dict(type='Accuracy2DHook'),
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
    dict(type='SaveCheckpointHook', interval=5, checkpoint_dir='./ckpt'),
]

parallel = dict(
    pipeline=dict(size=1),
    tensor=dict(size=4, mode='2d'),
)

fp16 = dict(mode=AMP_TYPE.PARALLEL, initial_scale=2 ** 8)

engine = dict(
    schedule=dict(num_microbatches=1)
)

logging = dict(root_path='./logs')

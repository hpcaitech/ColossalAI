import os
from pathlib import Path

BATCH_SIZE = 512
IMG_SIZE = 32
PATCH_SIZE = 4
DIM = 512
NUM_ATTENTION_HEADS = 8
NUM_CLASSES = 10
DEPTH = 6
LOG_NAME = 'vit1D_cifar10_tp=2_selfattention_V2'

# # ViT Base
# BATCH_SIZE = 512
# IMG_SIZE = 224
# PATCH_SIZE = 16
# DIM = 384
# NUM_ATTENTION_HEADS = 6
# NUM_CLASSES = 100
# DEPTH = 12
# LOG_NAME = 'vit1D_imagenet100'

train_data = dict(
    dataset=dict(
        type='CIFAR10Dataset',
        root=Path(os.environ['DATA']),
        download = True,
        transform_pipeline=[
            dict(type='RandomCrop', size=IMG_SIZE, padding=4),
            dict(type='RandomHorizontalFlip'),
            dict(type='ToTensor'),
            dict(type='Normalize',
                 mean=[0.4914, 0.4822, 0.4465],
                 std=[0.2023, 0.1994, 0.2010]),
        ]),
    dataloader=dict(batch_size=BATCH_SIZE,
                    pin_memory=True,
                    num_workers=4,
                    shuffle=True))

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
                 std=[0.2023, 0.1994, 0.2010]),
        ]),
    dataloader=dict(batch_size=400,
                    pin_memory=True,
                    num_workers=4,
                    shuffle=True))

optimizer = dict(type='Adam', lr=0.001, weight_decay=0)

loss = dict(type='CrossEntropyLoss1D', )

model = dict(
    type='VisionTransformerFromConfig',
    embedding_cfg=dict(
        type='ViTPatchEmbedding1D',
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=DIM,
    ),
    token_fusion_cfg=dict(type='ViTTokenFuser1D',
                          img_size=IMG_SIZE,
                          patch_size=PATCH_SIZE,
                          embed_dim=DIM,
                          drop_rate=0.1),
    norm_cfg=dict(
        type='LayerNorm',
        normalized_shape=DIM,
        eps=1e-6,
    ),
    block_cfg=dict(
        type='ViTBlock',
        attention_cfg=dict(
            type='ViTSelfAttention1DV2',
            hidden_size=DIM,
            num_attention_heads=NUM_ATTENTION_HEADS,
            attention_dropout_prob=0.,
            hidden_dropout_prob=0.1,
        ),
        droppath_cfg=dict(type='VanillaViTDropPath', ),
        mlp_cfg=dict(type='ViTMLP1D',
                     in_features=DIM,
                     dropout_prob=0.1,
                     mlp_ratio=1),
        norm_cfg=dict(
            type='LayerNorm',
            normalized_shape=DIM,
            eps=1e-6,
        ),
    ),
    head_cfg=dict(
        type='ViTHead1D',
        hidden_size=DIM,
        num_classes=NUM_CLASSES,
    ),
    embed_dim=DIM,
    depth=DEPTH,
    drop_path_rate=0.,
)

parallel = dict(
    pipeline=dict(size=1),
    tensor=dict(size=2, mode='1d'),
)

hooks = [
    dict(type='LogMetricByEpochHook'),
    # dict(type='LogTimingByEpochHook'),
    # dict(type='LogMemoryByEpochHook'),
    dict(type='TensorboardHook', log_dir=f'./tests/test_models/test_vision_transformer/test_vit_1d/tb_logs_{LOG_NAME}'),
    dict(
        type='Accuracy1DHook',
    ),
    dict(type='LossHook'),
    # dict(type='TensorboardHook', log_dir='./tfb_logs'),
    # dict(type='SaveCheckpointHook', interval=5, checkpoint_dir='./ckpt'),
    # dict(type='LoadCheckpointHook', epoch=20, checkpoint_dir='./ckpt')
]

logging = dict(
    root_path=f"./tests/test_models/test_vision_transformer/test_vit_1d/{LOG_NAME}"
)

lr_scheduler = dict(type='LinearWarmupLR', warmup_epochs=5)

num_epochs = 70

seed = 42
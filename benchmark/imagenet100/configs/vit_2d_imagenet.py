from colossalai.engine import AMP_TYPE

BATCH_SIZE = 128
LEARNING_RATE = 0.001
IMG_SIZE = 224
PATCH_SIZE = 16
DIM = 2048
NUM_ATTENTION_HEADS = 16
NUM_CLASSES = 1000
DEPTH = 48
NUM_EPOCHS = 300

parallel = dict(
    data=4,
    pipeline=1,
    tensor=dict(size=1, mode='2d'),
)

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
        attention_cfg=dict(type='ViTSelfAttention2D',
                           hidden_size=DIM,
                           num_attention_heads=NUM_ATTENTION_HEADS,
                           attention_dropout_prob=0.,
                           hidden_dropout_prob=0.1,
                           checkpoint=True),
        droppath_cfg=dict(type='VanillaViTDropPath', ),
        mlp_cfg=dict(type='ViTMLP2D',
                     in_features=DIM,
                     dropout_prob=0.1,
                     mlp_ratio=4,
                     checkpoint=True),
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

optimizer = dict(
    type='AdamW',
    lr=3e-3,
    weight_decay=0.3,
)

loss = dict(type='CrossEntropyLoss2D', reduction=True)

clip_grad = 1.0

num_epochs = NUM_EPOCHS

fp16 = dict(mode=AMP_TYPE.PARALLEL, initial_scale=2**8)

# this engine config can be ignored if you want to use default values
engine = dict(
    # schedule=None,
    schedule=dict(num_microbatches=4),
    gradient_handlers=None,
    gradient_accumulation=1,
    gradient_clipping=1.0,
)

hooks = [
    dict(type='LogMetricByEpochHook'),
    dict(type='LogMemoryByEpochHook'),
    dict(type='LogTimingByEpochHook'),
    dict(type='Accuracy2DHook'),
    dict(type='LossHook'),
    dict(type='LRSchedulerHook',
         by_epoch=True,
         lr_scheduler_cfg=dict(type='CosineAnnealingWarmupLR',
                               warmup_steps=32))
]

logging = dict(
    root_path=
    f"./vit_2d_imagenet1k_bs{BATCH_SIZE}_{fp16['mode']}_clip_grad{clip_grad}")

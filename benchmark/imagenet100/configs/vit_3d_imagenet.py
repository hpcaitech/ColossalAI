from colossalai.engine import AMP_TYPE

# VIT-S/16
IMG_SIZE = 224
PATCH_SIZE = 16
EMBED_SIZE = 384
HIDDEN_SIZE = 384
MLP_RATIO = 4
NUM_HEADS = 6
NUM_CLASSES = 100
DROP_RATE = 0.1
DEPTH = 12
###

# ### ViT-L/16
# IMG_SIZE = 224
# PATCH_SIZE = 16
# EMBED_SIZE = 10240
# HIDDEN_SIZE = 10240
# MLP_RATIO = 4
# NUM_HEADS = 64
# NUM_CLASSES = 1000
# DROP_RATE = 0.1
# DEPTH = 64
# ###

# # very large custom vit
# IMG_SIZE = 224
# PATCH_SIZE = 14
# EMBED_SIZE = 12288
# HIDDEN_SIZE = 12288
# MLP_RATIO = 4
# NUM_HEADS = 96
# NUM_CLASSES = 1000
# DROP_RATE = 0.1
# DEPTH = 96
# ###

BATCH_SIZE = 4096

TENSOR_PARALLEL = 8

parallel = dict(
    pipeline=1,
    tensor=dict(mode='3d', size=TENSOR_PARALLEL),
)

optimizer = dict(
    type='AdamW',
    lr=3e-3,
    weight_decay=0.3,
)

loss = dict(
    type='CrossEntropyLoss3D',
    label_smoothing=0.1,
)

model = dict(
    type='VisionTransformerFromConfig',
    embedding_cfg=dict(
        type='ViTPatchEmbedding3D',
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=3,
        embed_size=EMBED_SIZE,
        drop_prob=DROP_RATE,
        init_method='jax',
    ),
    block_cfg=dict(
        type='ViTBlock',
        norm_cfg=dict(
            type='LayerNorm3D',
            normalized_shape=HIDDEN_SIZE,
            eps=1e-6,
        ),
        attention_cfg=dict(
            type='ViTSelfAttention3D',
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=NUM_HEADS,
            attention_probs_dropout_prob=0.,
            hidden_dropout_prob=DROP_RATE,
            # checkpoint=True,
            init_method='jax',
        ),
        droppath_cfg=dict(type='VanillaViTDropPath', ),
        mlp_cfg=dict(
            type='ViTMLP3D',
            hidden_size=HIDDEN_SIZE,
            mlp_ratio=MLP_RATIO,
            hidden_dropout_prob=DROP_RATE,
            hidden_act='gelu',
            # checkpoint=True,
            init_method='jax',
        ),
    ),
    norm_cfg=dict(
        type='LayerNorm3D',
        normalized_shape=HIDDEN_SIZE,
        eps=1e-6,
    ),
    head_cfg=dict(
        type='ViTHead3D',
        in_features=HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        init_method='jax',
    ),
    embed_dim=HIDDEN_SIZE,
    depth=DEPTH,
    drop_path_rate=0.,
)

clip_grad = 1.0

engine = dict(
    schedule=None,
    gradient_handlers=None,
    gradient_accumulation=4,
    gradient_clipping=clip_grad,
)

num_epochs = 300

hooks = [
    dict(type='LogMetricByEpochHook'),
    # dict(type='LogMemoryByEpochHook'),
    # dict(type='LogTimingByEpochHook', ignore_num_train_steps=50),
    dict(type='Accuracy3DHook', ),
    dict(type='LossHook'),
    dict(type='LRSchedulerHook',
         by_epoch=True,
         lr_scheduler_cfg=dict(
             type='CosineAnnealingWarmupLR',
             warmup_steps=32,
         )),
]

# fp16 = dict(mode=AMP_TYPE.TORCH, )

logging = dict(
    root_path=
    f"./vit_3d_imagenet100_tp{TENSOR_PARALLEL}_bs{BATCH_SIZE}_clip_grad{clip_grad}")

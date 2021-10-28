import torch.nn as nn

IMG_SIZE = 224
DIM = 768
NUM_CLASSES = 1000
NUM_ATTN_HEADS = 12

model = dict(
    type='VisionTransformerFromConfig',
    embedding_cfg=dict(
        type='VanillaViTPatchEmbedding',
        img_size=IMG_SIZE,
        patch_size=16,
        in_chans=3,
        embed_dim=DIM
    ),
    norm_cfg=dict(
        type='LayerNorm',
        eps=1e-6,
        normalized_shape=DIM
    ),
    block_cfg=dict(
        type='ViTBlock',
        checkpoint=True,
        attention_cfg=dict(
            type='VanillaViTAttention',
            dim=DIM,
            num_heads=NUM_ATTN_HEADS,
            qkv_bias=True,
            attn_drop=0.,
            proj_drop=0.
        ),
        droppath_cfg=dict(
            type='VanillaViTDropPath',
        ),
        mlp_cfg=dict(
            type='VanillaViTMLP',
            in_features=DIM,
            hidden_features=DIM * 4,
            act_layer=nn.GELU,
            drop=0.
        ),
        norm_cfg=dict(
            type='LayerNorm',
            normalized_shape=DIM
        ),
    ),
    head_cfg=dict(
        type='VanillaViTHead',
        in_features=DIM,
        intermediate_features=DIM * 2,
        out_features=NUM_CLASSES
    ),
    depth=12,
    drop_path_rate=0.,
)

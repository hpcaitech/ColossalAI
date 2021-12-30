import math
from typing import Callable

import torch
from colossalai import nn as col_nn
from colossalai.nn.layer.utils import CheckpointModule
from colossalai.registry import LAYERS, MODELS
from torch import dtype, nn

__all__ = [
    'VisionTransformer',
    'vit_lite_depth7_patch4_32',
    'vit_tiny_patch4_32',
    'vit_tiny_patch16_224',
    'vit_tiny_patch16_384',
    'vit_small_patch16_224',
    'vit_small_patch16_384',
    'vit_small_patch32_224',
    'vit_small_patch32_384',
    'vit_base_patch16_224',
    'vit_base_patch16_384',
    'vit_base_patch32_224',
    'vit_base_patch32_384',
    'vit_large_patch16_224',
    'vit_large_patch16_384',
    'vit_large_patch32_224',
    'vit_large_patch32_384',
]

_init_rules = dict(
    torch=dict(
        embed=dict(
            weight_initializer=col_nn.init.kaiming_uniform_(a=math.sqrt(5)),
            bias_initializer=col_nn.init.xavier_uniform_(a=1, scale=1),
            position_embed_initializer=col_nn.init.zeros_(),
        ),
        transformer=dict(
            weight_initializer=col_nn.init.kaiming_uniform_(a=math.sqrt(5)),
            bias_initializer=col_nn.init.xavier_uniform_(a=1, scale=1),
        ),
        head=dict(
            weight_initializer=col_nn.init.kaiming_uniform_(a=math.sqrt(5)),
            bias_initializer=col_nn.init.xavier_uniform_(a=1, scale=1),
        ),
    ),
    jax=dict(
        embed=dict(
            weight_initializer=col_nn.init.lecun_normal_(),
            bias_initializer=col_nn.init.zeros_(),
            position_embed_initializer=col_nn.init.trunc_normal_(std=.02),
        ),
        transformer=dict(
            weight_initializer=col_nn.init.xavier_uniform_(),
            bias_initializer=col_nn.init.normal_(std=1e-6),
        ),
        head=dict(
            weight_initializer=col_nn.init.zeros_(),
            bias_initializer=col_nn.init.zeros_(),
        ),
    ),
)


@LAYERS.register_module
class ViTEmbedding(nn.Module):
    def __init__(self,
                 img_size: int,
                 patch_size: int,
                 in_chans: int,
                 embedding_dim: int,
                 dropout: float,
                 dtype: dtype = None,
                 flatten: bool = True,
                 init_method: str = 'torch'):
        super().__init__()
        self.patch_embed = col_nn.PatchEmbedding(img_size,
                                                 patch_size,
                                                 in_chans,
                                                 embedding_dim,
                                                 dtype=dtype,
                                                 flatten=flatten,
                                                 **_init_rules[init_method]['embed'])
        self.dropout = col_nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.dropout(x)
        return x


@LAYERS.register_module
class ViTSelfAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 attention_dropout: float,
                 dropout: float,
                 bias: bool = True,
                 dtype: dtype = None,
                 init_method: str = 'torch'):
        super().__init__()
        self.attention_head_size = dim // num_heads
        self.query_key_value = col_nn.Linear(dim,
                                             3 * dim,
                                             dtype=dtype,
                                             bias=bias,
                                             **_init_rules[init_method]['transformer'])
        self.attention_dropout = col_nn.Dropout(attention_dropout)
        self.dense = col_nn.Linear(dim, dim, dtype=dtype, bias=True, **_init_rules[init_method]['transformer'])
        self.dropout = col_nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        qkv = self.query_key_value(x)
        all_head_size = qkv.shape[-1] // 3
        num_attention_heads = all_head_size // self.attention_head_size
        new_qkv_shape = qkv.shape[:-1] + \
            (num_attention_heads, 3 * self.attention_head_size)
        qkv = qkv.view(new_qkv_shape)
        qkv = qkv.permute((0, 2, 1, 3))
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        x = torch.matmul(q, k.transpose(-1, -2))
        x = x / math.sqrt(self.attention_head_size)
        x = self.softmax(x)
        x = self.attention_dropout(x)

        x = torch.matmul(x, v)
        x = x.transpose(1, 2)
        new_context_layer_shape = x.size()[:-2] + (all_head_size, )
        x = x.reshape(new_context_layer_shape)

        x = self.dense(x)
        x = self.dropout(x)

        return x


@LAYERS.register_module
class ViTMLP(nn.Module):
    def __init__(self,
                 dim: int,
                 mlp_ratio: int,
                 activation: Callable,
                 dropout: float,
                 dtype: dtype = None,
                 bias: bool = True,
                 init_method: str = 'torch'):
        super().__init__()
        self.dense_1 = col_nn.Linear(dim,
                                     mlp_ratio * dim,
                                     dtype=dtype,
                                     bias=bias,
                                     **_init_rules[init_method]['transformer'])
        self.activation = activation
        self.dropout_1 = col_nn.Dropout(dropout)
        self.dense_2 = col_nn.Linear(mlp_ratio * dim,
                                     dim,
                                     dtype=dtype,
                                     bias=bias,
                                     **_init_rules[init_method]['transformer'])
        self.dropout_2 = col_nn.Dropout(dropout)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = self.dropout_2(x)
        return x


@LAYERS.register_module
class ViTHead(nn.Module):
    def __init__(self,
                 dim: int,
                 num_classes: int,
                 representation_size: int = None,
                 dtype: dtype = None,
                 bias: bool = True,
                 init_method: str = 'torch'):
        super().__init__()
        if representation_size:
            self.representation = col_nn.Linear(dim,
                                                representation_size,
                                                bias=bias,
                                                dtype=dtype,
                                                **_init_rules[init_method]['head'])
        else:
            self.representation = None
            representation_size = dim

        self.dense = col_nn.Classifier(representation_size,
                                       num_classes,
                                       dtype=dtype,
                                       bias=bias,
                                       **_init_rules[init_method]['head'])

    def forward(self, x):
        x = x[:, 0]
        if self.representation is not None:
            x = self.representation(x)
        x = self.dense(x)
        return x


@LAYERS.register_module
class ViTBlock(CheckpointModule):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int,
                 activation: Callable,
                 attention_dropout: float = 0.,
                 dropout: float = 0.,
                 drop_path: float = 0.,
                 layernorm_epsilon: float = 1e-6,
                 dtype: dtype = None,
                 bias: bool = True,
                 checkpoint: bool = False,
                 init_method: str = 'torch'):
        super().__init__(checkpoint)
        self.norm1 = col_nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
        self.attn = ViTSelfAttention(dim=dim,
                                     num_heads=num_heads,
                                     attention_dropout=attention_dropout,
                                     dropout=dropout,
                                     bias=bias,
                                     dtype=dtype,
                                     init_method=init_method)
        self.drop_path = col_nn.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = col_nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
        self.mlp = ViTMLP(dim=dim,
                          mlp_ratio=mlp_ratio,
                          activation=activation,
                          dropout=dropout,
                          dtype=dtype,
                          bias=bias,
                          init_method=init_method)

    def _forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@MODELS.register_module
class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 depth: int = 12,
                 num_heads: int = 12,
                 dim: int = 768,
                 mlp_ratio: int = 4,
                 attention_dropout: float = 0.,
                 dropout: float = 0.1,
                 drop_path: float = 0.,
                 layernorm_epsilon: float = 1e-6,
                 activation: Callable = nn.functional.gelu,
                 representation_size: int = None,
                 dtype: dtype = None,
                 bias: bool = True,
                 checkpoint: bool = False,
                 init_method: str = 'torch'):
        super().__init__()

        embed = ViTEmbedding(img_size=img_size,
                             patch_size=patch_size,
                             in_chans=in_chans,
                             embedding_dim=dim,
                             dropout=dropout,
                             dtype=dtype,
                             init_method=init_method)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        blocks = [
            ViTBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attention_dropout=attention_dropout,
                dropout=dropout,
                drop_path=dpr[i],
                activation=activation,
                dtype=dtype,
                bias=bias,
                checkpoint=checkpoint,
                init_method=init_method,
            ) for i in range(depth)
        ]

        norm = col_nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)

        head = ViTHead(dim=dim,
                       num_classes=num_classes,
                       representation_size=representation_size,
                       dtype=dtype,
                       bias=bias,
                       init_method=init_method)

        self.layers = nn.Sequential(
            embed,
            *blocks,
            norm,
            head,
        )

    def forward(self, x):
        x = self.layers(x)
        return x


def _create_vit_model(**model_kwargs):
    model = VisionTransformer(**model_kwargs)
    return model


@MODELS.register_module
def vit_lite_depth7_patch4_32(**kwargs):
    model_kwargs = dict(img_size=32, patch_size=4, dim=256, depth=7, num_heads=4, mlp_ratio=2, num_classes=10, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_tiny_patch4_32(**kwargs):
    model_kwargs = dict(img_size=32, patch_size=4, dim=512, depth=6, num_heads=8, mlp_ratio=1, num_classes=10, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_tiny_patch16_224(**kwargs):
    model_kwargs = dict(img_size=224, patch_size=16, dim=192, depth=12, num_heads=3, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_tiny_patch16_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=16, dim=192, depth=12, num_heads=3, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_small_patch16_224(**kwargs):
    model_kwargs = dict(img_size=224, patch_size=16, dim=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_small_patch16_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=16, dim=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_small_patch32_224(**kwargs):
    model_kwargs = dict(img_size=224, patch_size=32, dim=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_small_patch32_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=32, dim=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_base_patch16_224(**kwargs):
    model_kwargs = dict(img_size=224, patch_size=16, dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_base_patch16_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=16, dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_base_patch32_224(**kwargs):
    model_kwargs = dict(img_size=224, patch_size=32, dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_base_patch32_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=32, dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_large_patch16_224(**kwargs):
    model_kwargs = dict(img_size=224, patch_size=16, dim=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_large_patch16_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=16, dim=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_large_patch32_224(**kwargs):
    model_kwargs = dict(img_size=224, patch_size=32, dim=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_large_patch32_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=32, dim=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)

import torch
from torch import nn

from colossalai import nn as col_nn
from colossalai.context import ParallelMode
from colossalai.registry import MODELS

__all__ = [
    'VisionTransformer3D',
    'vit_tiny_1d_patch4_32',
    'vit_tiny_1d_patch16_224',
    'vit_tiny_1d_patch16_384',
    'vit_small_1d_patch16_224',
    'vit_small_1d_patch16_384',
    'vit_small_1d_patch32_224',
    'vit_small_1d_patch32_384',
    'vit_base_1d_patch16_224',
    'vit_base_1d_patch16_384',
    'vit_base_1d_patch32_224',
    'vit_base_1d_patch32_384',
    'vit_large_1d_patch16_224',
    'vit_large_1d_patch16_384',
    'vit_large_1d_patch32_224',
    'vit_large_1d_patch32_384',
]


class ViTBlock1D(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 hidden_dim: int,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = col_nn.ViTSelfAttention1D(dim, num_heads, attn_drop, drop)
        self.drop_path = col_nn.VanillaViTDropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = col_nn.ViTMLP1D(dim, 1, drop, 'gelu')

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@MODELS.register_module
class VisionTransformer1D(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 depth: int = 12,
                 num_heads: int = 12,
                 embed_dim: int = 768,
                 hidden_dim: int = 3072,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = col_nn.ViTPatchEmbedding1D(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            drop_rate,
        )

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            ViTBlock1D(embed_dim, num_heads, hidden_dim,
                       drop_rate, attn_drop_rate, dpr[i])
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, ParallelMode.PARALLEL_3D_INPUT,
                                       ParallelMode.PARALLEL_3D_WEIGHT)

        self.head = col_nn.ViTHead1D(hidden_dim, num_classes)
        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)
        return x


def _create_vit_model(**model_kwargs):
    model = VisionTransformer1D(**model_kwargs)
    return model


@MODELS.register_module
def vit_tiny_1d_patch4_32(**kwargs):
    model_kwargs = dict(img_size=32, patch_size=4, embed_dim=512,
                        depth=6, num_heads=8, hidden_dim=512, num_classes=10, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_tiny_1d_patch16_224(**kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=192,
                        depth=12, num_heads=3, hidden_dim=768, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_tiny_1d_patch16_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=16,
                        embed_dim=192, depth=12, num_heads=3, hidden_dim=768, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_small_1d_patch16_224(**kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384,
                        depth=12, num_heads=6, hidden_dim=1536, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_small_1d_patch16_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=16,
                        embed_dim=384, depth=12, num_heads=6, hidden_dim=1536, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_small_1d_patch32_224(**kwargs):
    model_kwargs = dict(patch_size=32, embed_dim=384,
                        depth=12, num_heads=6, hidden_dim=1536, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_small_1d_patch32_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=32,
                        embed_dim=384, depth=12, num_heads=6, hidden_dim=1536, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_base_1d_patch16_224(**kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768,
                        depth=12, num_heads=12, hidden_dim=3072, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_base_1d_patch16_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=16,
                        embed_dim=768, depth=12, num_heads=12, hidden_dim=3072, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_base_3d_patch32_224(**kwargs):
    model_kwargs = dict(patch_size=32, embed_dim=768,
                        depth=12, num_heads=12, hidden_dim=3072, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_base_1d_patch32_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=32,
                        embed_dim=768, depth=12, num_heads=12, hidden_dim=3072, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_large_3d_patch16_224(**kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=1024,
                        depth=24, num_heads=16, hidden_dim=4096, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_large_1d_patch16_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=16,
                        embed_dim=1024, depth=24, num_heads=16, hidden_dim=4096, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_large_1d_patch32_224(**kwargs):
    model_kwargs = dict(patch_size=32, embed_dim=1024,
                        depth=24, num_heads=16, hidden_dim=4096, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_large_1d_patch32_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=32,
                        embed_dim=1024, depth=24, num_heads=16, hidden_dim=4096, **kwargs)
    return _create_vit_model(**model_kwargs)

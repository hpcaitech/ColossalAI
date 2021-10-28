from colossalai.context import ParallelMode, seed
from colossalai import nn as clsl_nn
from colossalai.registry import MODELS
from torch import nn
import torch


__all__ = [
    'VisionTransformer2D',
    'vit_tiny_2d_patch4_32',
    'vit_tiny_2d_patch16_224',
    'vit_tiny_2d_patch16_384',
    'vit_small_2d_patch16_224',
    'vit_small_2d_patch16_384',
    'vit_small_2d_patch32_224',
    'vit_small_2d_patch32_384',
    'vit_base_2d_patch16_224',
    'vit_base_2d_patch16_384',
    'vit_base_2d_patch32_224',
    'vit_base_2d_patch32_384',
    'vit_large_2d_patch16_224',
    'vit_large_2d_patch16_384',
    'vit_large_2d_patch32_224',
    'vit_large_2d_patch32_384',
]


class ViTBlock2D(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: str = 'gelu'):
        super().__init__()
        self.norm1 = clsl_nn.LayerNorm2D(dim, eps=1e-6)
        self.attn = clsl_nn.ViTSelfAttention2D(dim, num_heads, attn_drop, drop)
        self.drop_path = clsl_nn.VanillaViTDropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.norm2 = clsl_nn.LayerNorm2D(dim, eps=1e-6)
        self.mlp = clsl_nn.ViTMLP2D(dim, mlp_ratio, act_layer, drop)

    def forward(self, x):
        y = self.attn(self.norm1(x))
        with seed(ParallelMode.TENSOR):
            x = x + self.drop_path(y)
        y = self.mlp(self.norm2(x))
        with seed(ParallelMode.TENSOR):
            x = x + self.drop_path(y)
        return x


@MODELS.register_module
class VisionTransformer2D(nn.Module):

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: int = 4,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 act_layer: str = 'gelu'):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = clsl_nn.ViTPatchEmbedding2D(
            img_size, patch_size, embed_dim, in_chans
        )

        self.splitter = clsl_nn.ViTInputSplitter2D()

        self.token_fuser = clsl_nn.ViTTokenFuser2D(
            img_size, patch_size, embed_dim, drop_rate
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            ViTBlock2D(embed_dim, num_heads, mlp_ratio, drop_rate,
                       attn_drop_rate, dpr[i], act_layer)
            for i in range(depth)
        ])

        self.norm = clsl_nn.LayerNorm2D(embed_dim, eps=1e-6)
        self.head = clsl_nn.ViTHead2D(self.num_features, num_classes) if num_classes > 0 \
            else nn.Identity()

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.splitter(x)
        x = self.token_fuser(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)
        return x


def _create_vit_model(**model_kwargs):
    model = VisionTransformer2D(**model_kwargs)
    return model


@MODELS.register_module
def vit_tiny_2d_patch4_32(**kwargs):
    model_kwargs = dict(img_size=32, patch_size=4, embed_dim=512,
                        depth=6, num_heads=8, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_tiny_2d_patch16_224(**kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=192,
                        depth=12, num_heads=3, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_tiny_2d_patch16_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=16, embed_dim=192,
                        depth=12, num_heads=3, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_small_2d_patch16_224(**kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384,
                        depth=12, num_heads=6, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_small_2d_patch16_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=16, embed_dim=384,
                        depth=12, num_heads=6, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_small_2d_patch32_224(**kwargs):
    model_kwargs = dict(patch_size=32, embed_dim=384,
                        depth=12, num_heads=6, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_small_2d_patch32_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=32, embed_dim=384,
                        depth=12, num_heads=6, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_base_2d_patch16_224(**kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768,
                        depth=12, num_heads=12, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_base_2d_patch16_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=16, embed_dim=768,
                        depth=12, num_heads=12, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_base_2d_patch32_224(**kwargs):
    model_kwargs = dict(patch_size=32, embed_dim=768,
                        depth=12, num_heads=12, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_base_2d_patch32_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=32, embed_dim=768,
                        depth=12, num_heads=12, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_large_2d_patch16_224(**kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=1024,
                        depth=24, num_heads=16, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_large_2d_patch16_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=16, embed_dim=1024,
                        depth=24, num_heads=16, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_large_2d_patch32_224(**kwargs):
    model_kwargs = dict(patch_size=32, embed_dim=1024,
                        depth=24, num_heads=16, **kwargs)
    return _create_vit_model(**model_kwargs)


@MODELS.register_module
def vit_large_2d_patch32_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=32, embed_dim=1024,
                        depth=24, num_heads=16, **kwargs)
    return _create_vit_model(**model_kwargs)
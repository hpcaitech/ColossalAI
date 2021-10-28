# modified from https://github.com/lucidrains/mlp-mixer-pytorch/blob/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py
from functools import partial
from colossalai.context import ParallelMode
from colossalai.registry import MODELS
from torch import nn
from colossalai import nn as col_nn
from colossalai.nn.layer.parallel_3d._utils import get_depth_from_env
from einops.layers.torch import Rearrange, Reduce

__all__ = [
    'MLPMixer',
]


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn, depth_3d):
        super().__init__()
        self.fn = fn
        self.norm = col_nn.LayerNorm3D(
            dim, depth_3d, ParallelMode.PARALLEL_3D_INPUT, ParallelMode.PARALLEL_3D_WEIGHT)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, depth_3d, expansion_factor=4, dropout=0., dense=None):
    if dense is None:
        dense = partial(col_nn.Linear3D, depth=depth_3d, input_parallel_mode=ParallelMode.PARALLEL_3D_INPUT,
                        weight_parallel_mode=ParallelMode.PARALLEL_3D_WEIGHT)
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )


@MODELS.register_module
def MLPMixer(image_size, channels, patch_size, dim, depth, num_classes, expansion_factor=4, dropout=0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size) ** 2
    depth_3d = get_depth_from_env()
    linear = partial(col_nn.Linear3D, depth=depth_3d, input_parallel_mode=ParallelMode.PARALLEL_3D_INPUT,
                     weight_parallel_mode=ParallelMode.PARALLEL_3D_WEIGHT)
    norm_layer = partial(col_nn.LayerNorm3D, depth=depth_3d, input_parallel_mode=ParallelMode.PARALLEL_3D_INPUT,
                         weight_parallel_mode=ParallelMode.PARALLEL_3D_WEIGHT)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                  p1=patch_size, p2=patch_size),
        linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(
                num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(
                dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        norm_layer(dim),
        Reduce('b n c -> b c', 'mean'),
        linear(dim, num_classes)
    )

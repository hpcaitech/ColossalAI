#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import torch
from torch import nn as nn

from colossalai.builder import build_layer
from colossalai.registry import LAYERS
from .._common_utils import to_2tuple


@LAYERS.register_module
class ViTBlock(nn.Module):
    """Vision Transformer block

    :param attention_cfg: config of attention layer
    :type attention_cfg: dict
    :param droppath_cfg: config of drop path
    :type droppath_cfg: dict
    :param mlp_cfg: config of MLP layer
    :type mlp_cfg: dict
    :param norm_cfg: config of normlization layer
    :type norm_cfg: dict
    """

    def __init__(self,
                 attention_cfg: dict,
                 droppath_cfg: dict,
                 mlp_cfg: dict,
                 norm_cfg: dict,
                 ):
        super().__init__()
        self.norm1 = build_layer(norm_cfg)
        self.attn = build_layer(attention_cfg)
        self.drop_path = build_layer(
            droppath_cfg) if droppath_cfg['drop_path'] > 0. else nn.Identity()
        self.norm2 = build_layer(norm_cfg)
        self.mlp = build_layer(mlp_cfg)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@LAYERS.register_module
class VanillaViTPatchEmbedding(nn.Module):
    """ 2D Image to Patch Embedding

    :param img_size: image size
    :type img_size: int
    :param patch_size: size of a patch
    :type patch_size: int
    :param in_chans: input channels
    :type in_chans: int
    :param embed_dim: embedding dimension
    :type embed_dim: int
    :param norm_layer: layer norm class, defaults to None
    :type norm_layer: Callable
    :param flattern: whether flatten the output
    :type flatten: bool
    :param drop: dropout rate
    :type drop: float
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None, flatten=True, drop=0.):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return x


@LAYERS.register_module
class VanillaViTMLP(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    :param in_features: input channels
    :type in_features: int
    :param hidden_features: channels of the output of the first dense layer
    :type hidden_features: int
    :param hidden_features: channels of the output of the second dense layer
    :type hidden_features: int
    :param act_layer: activation function
    :type act_layer: Callable
    :param drop: dropout rate
    :type drop: float

    """

    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    :param drop_prob: probability for dropout
    :type drop_prob: float
    :param training: whether it is training mode
    :type training: bool

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


@LAYERS.register_module
class VanillaViTDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    :param drop_prob: probability for dropout
    :type drop_path: float
    """

    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


@LAYERS.register_module
class VanillaViTAttention(nn.Module):
    """Vanilla attention layer of Vision Transformer

    :param dim: dimension of input tensor
    :type dim: int
    :param num_heads: number of attention heads
    :type num_heads: int, optional
    :param qkv_bias: enable bias for qkv if True, defaults to False
    :type qkv_bias: bool, optional
    :param attn_drop: dropout probability for attention layer, defaults to 0.
    :type attn_drop: float, optional
    :param proj_drop: dropout probability for linear layer, defaults to 0.
    :type proj_drop: float, optional
    """

    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


@LAYERS.register_module
class VanillaViTBlock(nn.Module):

    """Vanilla Vision Transformer block

    :param dim: dimension of input tensor
    :type dim: int
    :param num_heads: number of attention heads
    :type num_heads: int
    :param mlp_ratio: hidden size of MLP divided by embedding dim, defaults to 4.
    :type mlp_ratio: float, optional
    :param qkv_bias: enable bias for qkv if True, defaults to False
    :type qkv_bias: bool, optional
    :param drop: dropout probability, defaults to 0.
    :type drop: float, optional
    :param attn_drop: dropout probability for attention layer, defaults to 0.
    :type attn_drop: float, optional
    :param drop_path: drop path probability, defaults to 0.
    :type drop_path: float, optional
    :param act_layer: activation function, defaults to nn.GELU
    :type act_layer: torch.nn.Module, optional
    :param norm_layer: normalization layer, defaults to nn.LayerNorm
    :type norm_layer: torch.nn.Module, optional
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LAYERS.get_module('VanillaViTAttention')(dim,
                                                             num_heads=num_heads,
                                                             qkv_bias=qkv_bias,
                                                             attn_drop=attn_drop,
                                                             proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = LAYERS.get_module('VanillaViTDropPath')(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LAYERS.get_module('VanillaViTMLP')(in_features=dim,
                                                      hidden_features=mlp_hidden_dim,
                                                      act_layer=act_layer,
                                                      drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@LAYERS.register_module
class VanillaViTHead(nn.Module):
    """Output layer of vanilla Vision Transformer

    :param in_features: size of input tensor
    :type in_features: int
    :param intermediate_features: hidden size
    :type intermediate_features: int
    :param out_features: size of output tensor
    :type out_features: int
    :param bias: whether to add bias, defaults to True
    :type bias: bool, optional
    """

    def __init__(self,
                 in_features,
                 intermediate_features,
                 out_features,
                 bias=True
                 ):
        super().__init__()
        self.linear_1 = nn.Linear(
            in_features, intermediate_features, bias=bias)
        self.act = nn.Tanh()
        self.linear_2 = nn.Linear(
            intermediate_features, out_features, bias=bias)

    def forward(self, x):
        x = x[:, 0, :].squeeze(1)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x

#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from torch import nn as nn

from colossalai.builder import build_layer
from colossalai.registry import LAYERS


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

        # x_ = x
        # x_ = self.norm1(x_)
        # if self.checkpoint:
        #     x_ = checkpoint(self.attn, x_)
        # else:
        #     x_ = self.attn(x_)
        # x_ = self.drop_path(x_)
        # x = x + x_
        #
        # x_ = x
        # x_ = self.norm2(x_)
        # if self.checkpoint:
        #     x_ = checkpoint(self.mlp, x_)
        # else:
        #     x_ = self.mlp(x_)
        # x_ = self.drop_path(x_)
        # x = x + x_
        return x

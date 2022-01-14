#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from colossalai.registry import LAYERS
from colossalai.registry import MODELS
from colossalai.nn.model import ModelFromConfig


@MODELS.register_module
class VanillaResNet(ModelFromConfig):
    """ResNet from 
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """

    def __init__(
            self,
            num_cls: int,
            block_type: str,
            layers: List[int],
            norm_layer_type: str = 'BatchNorm2d',
            in_channels: int = 3,
            groups: int = 1,
            width_per_group: int = 64,
            zero_init_residual: bool = False,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            dilations=(1, 1, 1, 1)
    ) -> None:
        super().__init__()

        self.inplanes = 64
        self.zero_init_residual = zero_init_residual
        self.blocks = layers
        self.block_expansion = LAYERS.get_module(block_type).expansion
        self.dilations = dilations
        self.reslayer_common_cfg = dict(
            type='ResLayer',
            block_type=block_type,
            norm_layer_type=norm_layer_type,
            groups=groups,
            base_width=width_per_group
        )

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.layers_cfg = [
            # conv1
            dict(type='Conv2d',
                 in_channels=in_channels,
                 out_channels=self.inplanes,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 bias=False),
            # bn1
            dict(
                type=norm_layer_type,
                num_features=self.inplanes
            ),
            # relu
            dict(
                type='ReLU',
                inplace=True
            ),
            # maxpool
            dict(
                type='MaxPool2d',
                kernel_size=3,
                stride=2,
                padding=1
            ),
            # layer 1
            dict(
                inplanes=self.inplanes,
                planes=64,
                blocks=self.blocks[0],
                dilation=self.dilations[0],
                **self.reslayer_common_cfg
            ),
            # layer 2
            dict(
                inplanes=64 * self.block_expansion,
                planes=128,
                blocks=self.blocks[1],
                stride=2,
                dilate=replace_stride_with_dilation[0],
                dilation=self.dilations[1],
                **self.reslayer_common_cfg
            ),
            # layer  3
            dict(
                inplanes=128 * self.block_expansion,
                planes=256,
                blocks=layers[2],
                stride=2,
                dilate=replace_stride_with_dilation[1],
                dilation=self.dilations[2],
                **self.reslayer_common_cfg
            ),
            # layer 4
            dict(
                inplanes=256 * self.block_expansion,
                planes=512,
                blocks=layers[3], stride=2,
                dilate=replace_stride_with_dilation[2],
                dilation=self.dilations[3],
                **self.reslayer_common_cfg
            ),
            # avg pool
            dict(
                type='AdaptiveAvgPool2d',
                output_size=(1, 1)
            ),
            # flatten
            dict(
                type='LambdaWrapper',
                func=lambda mod, x: torch.flatten(x, 1)
            ),
            # linear
            dict(
                type='Linear',
                in_features=512 * self.block_expansion,
                out_features=num_cls
            )
        ]

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, LAYERS.get_module('ResNetBottleneck')):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, LAYERS.get_module('ResNetBasicBlock')):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

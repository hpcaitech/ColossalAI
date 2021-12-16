#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.nn as nn

from colossalai.registry import LAYERS
from .conv import conv1x1


@LAYERS.register_module
class ResLayer(nn.Module):

    def __init__(self,
                 block_type: str,
                 norm_layer_type: str,
                 inplanes: int,
                 planes: int,
                 blocks: int,
                 groups: int,
                 base_width: int,
                 stride: int = 1,
                 dilation: int = 1,
                 dilate: bool = False,
                 ):
        super().__init__()
        self.block = LAYERS.get_module(block_type)
        self.norm_layer = LAYERS.get_module(norm_layer_type)
        self.inplanes = inplanes
        self.planes = planes
        self.blocks = blocks
        self.groups = groups
        self.dilation = dilation
        self.base_width = base_width
        self.dilate = dilate
        self.stride = stride
        self.layer = self._make_layer()

    def _make_layer(self):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if self.dilate:
            self.dilation *= self.stride
            self.stride = 1
        if self.stride != 1 or self.inplanes != self.planes * self.block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, self.planes * self.block.expansion, self.stride),
                norm_layer(self.planes * self.block.expansion),
            )

        layers = []
        layers.append(self.block(self.inplanes, self.planes, self.stride, downsample, self.groups,
                                 self.base_width, previous_dilation, norm_layer))
        self.inplanes = self.planes * self.block.expansion
        for _ in range(1, self.blocks):
            layers.append(self.block(self.inplanes, self.planes, groups=self.groups,
                                     base_width=self.base_width, dilation=self.dilation,
                                     norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

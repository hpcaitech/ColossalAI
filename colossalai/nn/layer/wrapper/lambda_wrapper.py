#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.nn as nn

from colossalai.builder import build_layer
from colossalai.registry import LAYERS


@LAYERS.register_module
class LambdaWrapper(nn.Module):
    """Wrap a function to nn.Module, which takes a config of layers and can fully access them

    :param func: User customed function
    :type func: Callable
    :param layers_cfg: Config of layers, defaults to None
    :type layers_cfg: dict, optional
    """

    def __init__(self, func, layers_cfg: dict = None):
        super().__init__()
        self.func = func
        self.layers = self._build_layers(layers_cfg)

    def _build_layers(self, layers_cfg: dict):
        if layers_cfg is None:
            return None
        else:
            layers = []

            for cfg in layers_cfg:
                layer = build_layer(cfg)
                layers.append(layer)
            return layers

    def forward(self, *args, **kwargs):
        return self.func(self, *args, **kwargs)

#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from abc import ABC, abstractmethod

import torch.nn as nn

from colossalai.builder import build_layer


class ModelFromConfig(nn.Module, ABC):

    def __init__(self):
        super(ModelFromConfig, self).__init__()
        self.layers = nn.ModuleList()
        self.layers_cfg = []

    def build_from_cfg(self, start=None, end=None):
        assert hasattr(self, 'layers_cfg'), 'Cannot find attribute layers_cfg from the module, please check the ' \
                                            'spelling and if you have initialized this variable'
        if start is None:
            start = 0
        if end is None:
            end = len(self.layers_cfg)
        for cfg in self.layers_cfg[start: end]:
            layer = build_layer(cfg)
            self.layers.append(layer)

    @abstractmethod
    def init_weights(self):
        pass

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """Use this function to override the state dict for
        saving checkpoints."""
        return self.state_dict(destination, prefix, keep_vars)

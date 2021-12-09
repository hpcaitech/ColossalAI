#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch

from colossalai.registry import MODELS
from colossalai.nn.model.model_from_config import ModelFromConfig


@MODELS.register_module
class VisionTransformerFromConfig(ModelFromConfig):
    """Vision Transformer from 
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/pdf/2010.11929>`_.

    """

    def __init__(self,
                 embedding_cfg: dict,
                 norm_cfg: dict,
                 block_cfg: dict,
                 head_cfg: dict,
                 token_fusion_cfg: dict = None,
                 embed_dim=768,
                 depth=12,
                 drop_path_rate=0.,
                 tensor_splitting_cfg: dict = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = 1
        self.tensor_splitting_cfg = tensor_splitting_cfg
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        if token_fusion_cfg is None:
            token_fusion_cfg = []
        else:
            token_fusion_cfg = [token_fusion_cfg]

        self.layers_cfg = [
            embedding_cfg,

            # input tensor splitting
            *self._generate_tensor_splitting_cfg(),
            *token_fusion_cfg,

            # blocks
            *self._generate_block_cfg(
                dpr=dpr, block_cfg=block_cfg, depth=depth),

            # norm
            norm_cfg,

            # head
            head_cfg
        ]

    def _fuse_tokens(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        return x

    def _generate_block_cfg(self, dpr, depth, block_cfg):
        blocks_cfg = []

        for i in range(depth):
            _cfg = block_cfg.copy()
            _cfg['droppath_cfg']['drop_path'] = dpr[i]
            blocks_cfg.append(_cfg)

        return blocks_cfg

    def _generate_tensor_splitting_cfg(self):
        if self.tensor_splitting_cfg:
            return [self.tensor_splitting_cfg]
        else:
            return []

    def forward(self, x):  # [512, 3, 32, 32]
        for layer in self.layers:
            if isinstance(x, tuple):
                x = layer(*x)
            else:
                x = layer(x)
        return x  # [256, 5]

    def init_weights(self):
        # TODO: add init weights
        pass

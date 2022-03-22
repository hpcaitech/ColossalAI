#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.nn as nn

from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc


class ParallelLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.data_parallel_rank = 0 if not gpc.is_initialized(ParallelMode.DATA) else gpc.get_local_rank(
            ParallelMode.DATA)
        self.data_parallel_size = 1 if not gpc.is_initialized(ParallelMode.DATA) else gpc.get_world_size(
            ParallelMode.DATA)

        self.tensor_parallel_rank = 0 if not gpc.is_initialized(ParallelMode.TENSOR) else gpc.get_local_rank(
            ParallelMode.TENSOR)
        self.tensor_parallel_size = 1 if not gpc.is_initialized(ParallelMode.TENSOR) else gpc.get_world_size(
            ParallelMode.TENSOR)

        self.pipeline_parallel_rank = 0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(
            ParallelMode.PIPELINE)
        self.pipeline_parallel_size = 1 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_world_size(
            ParallelMode.PIPELINE)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)
        if gpc.get_local_rank(ParallelMode.TENSOR) != 0:
            missing_keys.clear()
            unexpected_keys.clear()

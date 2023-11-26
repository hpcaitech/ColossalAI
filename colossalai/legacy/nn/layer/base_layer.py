#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from contextlib import contextmanager

import torch.nn as nn

from colossalai.legacy.context import ParallelMode
from colossalai.legacy.core import global_context as gpc


class ParallelLayer(nn.Module):
    global_state_dict: bool = True

    def __init__(self):
        super().__init__()
        self.data_parallel_rank = (
            0 if not gpc.is_initialized(ParallelMode.DATA) else gpc.get_local_rank(ParallelMode.DATA)
        )
        self.data_parallel_size = (
            1 if not gpc.is_initialized(ParallelMode.DATA) else gpc.get_world_size(ParallelMode.DATA)
        )

        self.tensor_parallel_rank = (
            0 if not gpc.is_initialized(ParallelMode.TENSOR) else gpc.get_local_rank(ParallelMode.TENSOR)
        )
        self.tensor_parallel_size = (
            1 if not gpc.is_initialized(ParallelMode.TENSOR) else gpc.get_world_size(ParallelMode.TENSOR)
        )

        self.pipeline_parallel_rank = (
            0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(ParallelMode.PIPELINE)
        )
        self.pipeline_parallel_size = (
            1 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_world_size(ParallelMode.PIPELINE)
        )

    def _load_from_global_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _save_to_global_state_dict(self, destination, prefix, keep_vars):
        return super()._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.global_state_dict:
            if gpc.get_local_rank(ParallelMode.TENSOR) != 0:
                missing_keys.clear()
                unexpected_keys.clear()
            return self._load_from_global_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if self.global_state_dict:
            return self._save_to_global_state_dict(destination, prefix, keep_vars)
        return super()._save_to_state_dict(destination, prefix, keep_vars)

    @classmethod
    @contextmanager
    def use_local_state_dict(cls):
        try:
            cls.global_state_dict = False
            yield
        finally:
            cls.global_state_dict = True

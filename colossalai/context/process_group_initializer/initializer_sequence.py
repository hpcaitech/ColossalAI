#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from colossalai.registry import DIST_GROUP_INITIALIZER
from .initializer_tensor import Initializer_Tensor
from .process_group_initializer import ProcessGroupInitializer
from ..parallel_mode import ParallelMode


@DIST_GROUP_INITIALIZER.register_module
class Initializer_Sequence(ProcessGroupInitializer):
    '''A ProcessGroupInitializer for sequence parallelism.
    '''

    def __init__(self,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # reuse tensor parallel code
        self._initializer = Initializer_Tensor(*args, **kwargs)

    def init_dist_group(self):
        local_rank, group_world_size, process_group, ranks_in_group, mode = self._initializer.init_dist_group()

        # change mode to sequence
        mode = ParallelMode.SEQUENCE
        
        return local_rank, group_world_size, process_group, ranks_in_group, mode

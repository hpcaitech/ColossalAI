#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from abc import ABC, abstractmethod

from colossalai.context import Config


class ProcessGroupInitializer(ABC):
    '''An object, knowing the parallelism configuration, that initializes parallel groups.
    '''
    def __init__(self,
                 rank: int,
                 world_size: int,
                 config: Config,
                 data_parallel_size: int,
                 pipeline_parlalel_size: int,
                 tensor_parallel_size: int
                 ):
        self.rank = rank
        self.world_size = world_size
        self.data_parallel_size = data_parallel_size
        self.config = config
        self.pipeline_parallel_size = pipeline_parlalel_size
        self.tensor_parallel_size = tensor_parallel_size
        super().__init__()

    @abstractmethod
    def init_dist_group(self):
        pass

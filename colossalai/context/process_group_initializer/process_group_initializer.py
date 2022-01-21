#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from abc import ABC, abstractmethod

from colossalai.context import Config


class ProcessGroupInitializer(ABC):
    """An object, knowing the parallelism configuration, that initializes parallel groups.

    :param rank: The rank of current process
    :param world_size: Size of whole communication world
    :param config: Running configuration
    :param data_parallel_size: Size of data parallel
    :param pipeline_parallel_size: Size of pipeline parallel
    :param tensor_parallel_size: Size of tensor parallel

    :type rank: int
    :type world_size: int
    :type config: Config
    :type data_parallel_size: int
    :type pipeline_parallel_size: int
    :type tensor_parallel_size: int
    """
    def __init__(self,
                 rank: int,
                 world_size: int,
                 config: Config,
                 data_parallel_size: int,
                 pipeline_parallel_size: int,
                 tensor_parallel_size: int
                 ):
        self.rank = rank
        self.world_size = world_size
        self.data_parallel_size = data_parallel_size
        self.config = config
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        super().__init__()

    @abstractmethod
    def init_dist_group(self):
        pass

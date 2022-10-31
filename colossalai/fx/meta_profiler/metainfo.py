from typing import Callable

import numpy as np
import torch

from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    MemoryCost,
    OperationData,
    OperationDataType,
    ShardingStrategy,
    StrategiesVector,
    TrainCycleItem,
)
from colossalai.tensor.sharding_spec import ShardingSpec

from .registry import meta_register

__all__ = ['MetaInfo']


class MetaInfo:
    """MetaInfo class
    This class is used to store meta info based on sharding strategy and the given
    target function.
    """

    def __init__(self, strategy: ShardingStrategy = None, target: Callable = None) -> None:
        # compute cost of forward and backward computation
        self.compute_cost: TrainCycleItem

        # compute memory cost of forward and backward phase
        self.memory_cost: TrainCycleItem

        # list of input tensors
        self.fwd_in: list[OperationData]

        # sharding strategy
        self._strategy = strategy

        # target function
        self._target = target

        # compute metainfo if possible
        if self._strategy is not None and self._target is not None:
            self.compute_metainfo()

    @property
    def strategy(self) -> ShardingStrategy:
        return self._strategy

    @property
    def target(self) -> Callable:
        return self._target

    @strategy.setter
    def strategy(self, strategy: ShardingStrategy) -> None:
        self._strategy = strategy
        if self._strategy is not None and self._target is not None:
            self.compute_metainfo()

    @target.setter
    def target(self, target: Callable) -> None:
        self._target = target
        if self._strategy is not None and self._target is not None:
            self.compute_metainfo()

    def compute_sharded_tensor(self, operation_data: OperationData, sharding_spec: ShardingSpec) -> torch.Tensor:
        """
        Compute sharded meta tensor based on the given data and sharding spec.
        """
        shard_sequnce = sharding_spec.sharding_sequence
        device_mesh = sharding_spec.device_mesh
        shape = operation_data.data.shape

        new_shape = []
        for dim, shard in zip(shape, shard_sequnce):
            if shard.is_replica:
                # replica
                new_shape.append(dim)
            else:
                # sharded according to device_mesh shape
                new_shape.append(dim // np.prod(np.array([device_mesh.mesh_shape[i] for i in shard.shard_list])))

        return OperationData(name=operation_data.name,
                             data=torch.zeros(new_shape, device="meta"),
                             type=operation_data.type,
                             logical_shape=operation_data.logical_shape)

    def compute_metainfo(self):
        """
        Compute meta info based on sharding strategy and the given target function.
        """

        assert meta_register.has(self._target), f'{self._target} not found in the meta registry'
        meta_func = meta_register.get(self._target)

        # construct args for meta_func
        args = [self.compute_sharded_tensor(k, v) for k, v in self._strategy.sharding_specs.items()]

        # compute metainfo with meta_func
        self.compute_cost, self.memory_cost, self.fwd_in = meta_func(*args)

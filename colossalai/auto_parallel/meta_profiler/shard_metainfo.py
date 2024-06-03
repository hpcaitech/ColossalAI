from typing import Callable, List

import torch

from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, ShardingStrategy, TrainCycleItem
from colossalai.tensor.sharding_spec import ShardingSpec

from .constants import INPLACE_MODULE, INPLACE_OPS, NO_SAVE_ACTIVATION
from .registry import meta_register

__all__ = ["ShardMetaInfo"]


class ShardMetaInfo:
    """ShardMetaInfo class
    This class is used to store meta info based on sharding strategy and the given
    target function.
    """

    def __init__(self, strategy: ShardingStrategy = None, target: Callable = None) -> None:
        # compute cost of forward and backward computation
        self.compute_cost: TrainCycleItem

        # compute memory cost of forward and backward phase
        self.memory_cost: TrainCycleItem

        # list of input tensors
        self.fwd_in: List[torch.Tensor]

        # list of buffer tensors
        self.fwd_buffer: List[torch.Tensor]

        # list of output tensors
        self.fwd_out: List[torch.Tensor]

        # sharding strategy
        self._strategy = strategy

        # target function
        self._target = target

        # compute shard_metainfo if possible
        if self._strategy is not None and self._target is not None:
            self.compute_shard_metainfo()

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
            self.compute_shard_metainfo()

    @target.setter
    def target(self, target: Callable) -> None:
        self._target = target
        if self._strategy is not None and self._target is not None:
            self.compute_shard_metainfo()

    def compute_sharded_opdata(self, operation_data: OperationData, sharding_spec: ShardingSpec):
        """
        Compute sharded opdata based on the given data and sharding spec.
        """

        if isinstance(sharding_spec, ShardingSpec):
            op_data = OperationData(
                name=operation_data.name,
                data=torch.zeros(sharding_spec.get_sharded_shape_per_device(), device="meta"),
                type=operation_data.type,
                logical_shape=operation_data.logical_shape,
            )
        elif isinstance(sharding_spec, (list, tuple)):
            data = operation_data.data
            assert isinstance(data, (list, tuple)), f"Data Should be list or tuple, but got {type(data)}."
            assert len(data) == len(sharding_spec), f"Length of data and sharding spec should be the same."
            sharded_data = []
            for d, s in zip(data, sharding_spec):
                sharded_data.append(torch.zeros(s.get_sharded_shape_per_device(), device="meta"))
            op_data = OperationData(name=operation_data.name, data=sharded_data, type=operation_data.type)
        else:
            raise ValueError(f"Sharding spec should be ShardingSpec or list, but got {type(sharding_spec)}.")

        return op_data

    def compute_shard_metainfo(self):
        """
        Compute meta info based on sharding strategy and the given target function.
        """
        assert meta_register.has(self._target.__class__) or meta_register.has(
            self._target
        ), f"Meta info for {self._target} is not registered."
        if meta_register.has(self._target.__class__):
            # module
            meta_func = meta_register.get(self._target.__class__)

            # check whether the target in the list that we don't need to save activation
            save_fwd_in = self._target.__class__ not in NO_SAVE_ACTIVATION
        else:
            # function
            meta_func = meta_register.get(self._target)

            # check whether the target in the list that we don't need to save activation
            save_fwd_in = self._target.__class__ not in NO_SAVE_ACTIVATION

        # construct args for meta_func
        args = [self.compute_sharded_opdata(k, v) for k, v in self._strategy.sharding_specs.items()]

        # construct kwargs
        if self.target in INPLACE_MODULE:
            kwargs = {"inplace": self.target.inplace}
        elif self.target in INPLACE_OPS:
            kwargs = {"inplace": True}
        else:
            kwargs = {"inplace": False}

        # compute metainfo with meta_func
        self.compute_cost, self.memory_cost, self.fwd_in, self.fwd_buffer, self.fwd_out = meta_func(*args, **kwargs)

        # process corner case for NO_SAVE_ACTIVATION
        if not save_fwd_in:
            self.fwd_in = []

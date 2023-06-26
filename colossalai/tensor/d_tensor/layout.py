import operator
from functools import reduce

import torch

from colossalai.device.device_mesh import DeviceMesh

from .misc import DuplicatedShardingDimensionError, ShardingNotDivisibleError
from .sharding_spec import ShardingSpec


class Layout:
    """Layout of a tensor.

    Attributes:
        device_mesh: the device mesh to store the tensor distributed.
        sharding_spec: the sharding specification to describe how the tensor is sharded.
        global_shape: the entire shape of the global tensor.
    """

    def __init__(self, device_mesh: DeviceMesh, sharding_spec: ShardingSpec, global_shape: torch.Size):
        self.device_mesh = device_mesh
        self.sharding_spec = sharding_spec
        self.global_shape = global_shape
        self._sanity_check()

    def __hash__(self) -> int:
        return hash(f'{self.sharding_spec}')

    def get_sharded_shape_per_device(self):
        sharded_shape = list(self.global_shape)
        for dim, shard_list in self.sharding_spec.dim_partition_dict.items():
            mesh_list = [self.device_mesh.shape[mesh_dim] for mesh_dim in shard_list]
            shard_partitions = reduce(operator.mul, mesh_list, 1)
            assert sharded_shape[
                dim] % shard_partitions == 0, f'Cannot shard dimension {dim} into {shard_partitions} partitions.'
            sharded_shape[dim] //= shard_partitions
        return torch.Size(sharded_shape)

    def _sanity_check(self):
        sharding_spec = self.sharding_spec

        # make sure all axes in logical device mesh only be used once
        if self.device_mesh.logical_mesh_id is not None:
            dim_check_list = list(range(self.device_mesh.logical_mesh_id.dim()))
            for dim, shard_list in sharding_spec.dim_partition_dict.items():
                for element in shard_list:
                    if element in dim_check_list:
                        dim_check_list.remove(element)
                    else:
                        raise DuplicatedShardingDimensionError(
                            f"find an invalid sharding axis {element} in dim_partition_dict in tensor dimension {dim}.")

        # make sure that the sharding for a dimension is divisible by the number of devices
        for dim, shard_list in sharding_spec.dim_partition_dict.items():
            tensor_dim_size = self.global_shape[dim]
            num_devices = 1

            for element in shard_list:
                num_devices *= self.device_mesh.shape[element]

            if tensor_dim_size % num_devices != 0:
                raise ShardingNotDivisibleError(
                    f'The size of dimension at index {dim} is {tensor_dim_size}, it cannot be sharded over {num_devices} devices.'
                )

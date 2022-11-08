import operator
from copy import deepcopy
from functools import reduce

import torch

from colossalai.device.device_mesh import DeviceMesh

from .utils import merge_same_dim_mesh_list

__all__ = ['_DimSpec', 'ShardingException', 'ShardingSpec']

ALLGATHER_COST = 20
SHARD_COST = 5
STEP_PENALTY = 6
NAN = 'nan'


class _DimSpec:
    '''
    Sharding spec for single dimension of the sharded tensor decribe the sharding dimension of
    logical device mesh and give a method to compute the difference between them.
    This class is used internally in ShardingSpec.

    Argument:
        shard_list(List[int]): if shard_list is None, the dim spec will be 'R' type.
            Otherwise, the element in shard_list means the data will be sharded in that dimension.
    '''

    def __init__(self, shard_list):
        self.is_replica = len(shard_list) == 0
        self.shard_list = shard_list
        self.build_difference_2d_dict()

    def __eq__(self, other):
        return str(self) == str(other)

    def __repr__(self):
        if self.is_replica:
            return 'R'
        target = 'S'
        for dim in self.shard_list:
            target += str(dim)
        return target

    def _convert_str_to_shard_list(self, str_spec):
        '''
        Conver str_spec into shard_list.

        Argument:
            str_spec(str): dim spec in str type.
        '''

        if str_spec == 'R':
            return []
        if str_spec == 'S0':
            return [0]
        if str_spec == 'S1':
            return [1]
        if str_spec == 'S01':
            return [0, 1]

    def build_difference_2d_dict(self):
        '''
        Build a difference maping for 2D device mesh case. It will be used to
        compute the difference between DimSpec pairs.
        '''

        source_spec_list = ['R', 'S0', 'S1', 'S01']
        target_spec_list = ['R', 'S0', 'S1', 'S01']
        difference_dict = {}
        for source_spec in source_spec_list:
            for target_spec in target_spec_list:
                legal_sharding_dims = []
                spec_pair = (deepcopy(source_spec), deepcopy(target_spec))
                source_shard_list = self._convert_str_to_shard_list(source_spec)
                target_shard_list = self._convert_str_to_shard_list(target_spec)

                # source same as target
                if source_shard_list == target_shard_list:
                    difference = 0

                # all_gather(source) -> target
                elif len(source_shard_list
                        ) == len(target_shard_list) + 1 and source_shard_list[:-1] == target_shard_list:
                    difference = ALLGATHER_COST

                # shard(source) -> target
                elif len(source_shard_list) == len(
                        target_shard_list) - 1 and source_shard_list == target_shard_list[:-1] and target_shard_list[
                            -1] not in source_shard_list:
                    difference = SHARD_COST

                # S1 -> S0 or S0 -> S1
                elif len(source_shard_list) == len(target_shard_list):
                    # source -> R -> target
                    difference = ALLGATHER_COST + STEP_PENALTY + SHARD_COST

                # R -> S01
                elif len(source_shard_list) == len(target_shard_list) - 2:
                    difference = SHARD_COST + STEP_PENALTY + SHARD_COST

                # S01 -> R
                elif len(source_shard_list) == len(target_shard_list) + 2:
                    difference = ALLGATHER_COST + STEP_PENALTY + ALLGATHER_COST

                # S1 -> S01
                elif len(source_shard_list) == len(target_shard_list) - 1:
                    difference = ALLGATHER_COST + STEP_PENALTY + SHARD_COST + STEP_PENALTY + SHARD_COST

                # S01 -> S1
                elif len(source_shard_list) == len(target_shard_list) + 1:
                    difference = ALLGATHER_COST + STEP_PENALTY + ALLGATHER_COST + STEP_PENALTY + SHARD_COST

                else:
                    difference = NAN
                difference_dict[spec_pair] = difference

        self.difference_dict = difference_dict

    def difference(self, other):
        '''
        The difference between two _DimSpec.

        Argument:
            other(_DimSpec): the dim spec to compare with.

        Return:
            difference(int): the difference between two _DimSpec.

        Example:
            dim_spec = _DimSpec([0])
            other_dim_spec = _DimSpec([0, 1])
            print(dim_spec.difference(other_dim_spec))

        Output:
            5
        '''
        difference = self.difference_dict[(str(self), str(other))]
        return difference


class ShardingSpecException(Exception):
    pass


class ShardingOutOfIndexError(ShardingSpecException):
    pass


class DuplicatedShardingDimensionError(ShardingSpecException):
    pass


class ShardingNotDivisibleError(ShardingSpecException):
    pass


class ShardingSpec:
    '''
    Sharding spec for a tensor, it contains info of the logical device mesh this tensor belong
    to, the entire shape of the tensor before sharded, and the sharding sequence looks like
    [R, R, S0, S1].

    Argument:
        device_mesh(DeviceMesh): A logical view of a physical mesh.
        entire_shape(torch.Size): The entire shape of tensor before sharded.
        dim_partition_dict(Dict[int, List[int]]， optional): The key is the dimension of tensor to be sharded,
            and the value of the key decribe which logical axis will be sharded in that dimension.
        sharding_sequence(List[_DimSpec], optional): A straight view of ShardingSpec looks like [R, R, S0, S1].
    '''

    def __init__(self,
                 device_mesh: DeviceMesh,
                 entire_shape: torch.Size,
                 dim_partition_dict=None,
                 sharding_sequence=None):
        self.device_mesh = device_mesh

        if isinstance(entire_shape, (list, tuple)):
            entire_shape = torch.Size(entire_shape)
        self.entire_shape = entire_shape
        self.dim_partition_dict = dim_partition_dict
        self.sharding_sequence = sharding_sequence
        if self.sharding_sequence is None:
            assert self.dim_partition_dict is not None, f'dim_partition_dict should not be None, if sharding_sequence is NoneType object.'
            self.dim_partition_dict = merge_same_dim_mesh_list(dim_size=len(entire_shape),
                                                               dim_partition_dict=self.dim_partition_dict)
            self.convert_dict_to_shard_sequence()
        elif self.dim_partition_dict is None:
            assert self.sharding_sequence is not None, f'sharding_sequence should not be None, if dim_partition_dict is NoneType object.'
            self.convert_shard_sequence_to_dict()
        self._sanity_check()

    def __repr__(self):
        res_list = ["DistSpec:"]
        res_list.append(f"\n\tshard_sequence: " + ",".join(str(dimspec) for dimspec in self.sharding_sequence))
        res_list.append(f"\n\tdevice_mesh_shape: {self.device_mesh.mesh_shape}")
        return ' '.join(res_list)

    def _sanity_check(self):
        # make sure all axes in logical device mesh only be used once
        dim_check_list = list(range(self.device_mesh.logical_mesh_id.dim()))
        for dim, shard_list in self.dim_partition_dict.items():
            for element in shard_list:
                if element in dim_check_list:
                    dim_check_list.remove(element)
                else:
                    raise DuplicatedShardingDimensionError(
                        f"find an invalid sharding axis {element} in dim_partition_dict in tensor dimension {dim}.")

        # make sure that the dimension is not out of index
        for dim in self.dim_partition_dict.keys():
            if dim >= len(self.entire_shape):
                raise ShardingOutOfIndexError(
                    f"The dim_partition_dict specifies to shard dimension {dim} but the entire_shape only has {len(self.entire_shape)} dimensions"
                )

        # make sure that the sharding for a dimension is divisible by the number of devices
        for dim, shard_list in self.dim_partition_dict.items():
            tensor_dim_size = self.entire_shape[dim]
            num_devices = 1

            for element in shard_list:
                num_devices *= self.device_mesh.mesh_shape[element]

            if tensor_dim_size % num_devices != 0:
                raise ShardingNotDivisibleError(
                    f'The size of dimension at index {dim} is {tensor_dim_size}, it cannot be sharded over {num_devices} devices.'
                )

    def convert_dict_to_shard_sequence(self):
        '''
        Convert dim_partition_dict into list of _DimSpec, and assign it to sharding_sequence.
        '''
        sharding_sequence = [_DimSpec([])] * len(self.entire_shape)
        for dim, shard_list in self.dim_partition_dict.items():
            sharding_sequence[dim] = _DimSpec(shard_list)
        self.sharding_sequence = sharding_sequence

    def convert_shard_sequence_to_dict(self):
        '''
        Convert sharding_sequence into dim_partition_dict.
        '''
        new_dim_partition_dict = {}
        for index, dim_spec in enumerate(self.sharding_sequence):
            if not dim_spec.is_replica:
                if index not in new_dim_partition_dict:
                    new_dim_partition_dict[index] = []
                new_dim_partition_dict[index].extend(dim_spec.shard_list)
        self.dim_partition_dict = new_dim_partition_dict

    def sharding_sequence_difference(self, other):
        '''
        This function is a naive version of difference computation. It just simply accumulates difference every dimension between the
        pair of sharding sequence.

        Example:
            dim_partition_dict = {0: [0, 1]}
            # DistSpec:
            #     shard_sequence: S01,R,R
            #     device_mesh_shape: (4, 4)
            sharding_spec = ShardingSpec(device_mesh, entire_shape, dim_partition_dict)
            dim_partition_dict_to_compare = {0: [0], 1: [1]}
            # DistSpec:
            #     shard_sequence: S0,S1,R
            #     device_mesh_shape: (4, 4)
            sharding_spec_to_compare = ShardingSpec(device_mesh, entire_shape, dim_partition_dict_to_compare)
            print(sharding_spec.sharding_sequence_difference(sharding_spec_to_compare))

        Output:
            25

        Argument:
            other(ShardingSpec): The ShardingSpec to compared with.

        Return:
            difference(int): Difference between two ShardingSpec.
        '''
        assert len(self.sharding_sequence) == len(
            other.sharding_sequence), f'Cannot compare difference for two sharding specs with different length.'
        difference = 0
        for orig_dim_spec, other_dim_spec in zip(self.sharding_sequence, other.sharding_sequence):
            difference += orig_dim_spec.difference(other_dim_spec)
        return difference

    def get_sharded_shape_per_device(self):

        sharded_shape = list(self.entire_shape)
        for dim, shard_list in self.dim_partition_dict.items():
            mesh_list = [self.device_mesh.mesh_shape[mesh_dim] for mesh_dim in shard_list]
            shard_partitions = reduce(operator.mul, mesh_list, 1)
            assert sharded_shape[
                dim] % shard_partitions == 0, f'Cannot shard dimension {dim} into {shard_partitions} partitions.'
            sharded_shape[dim] //= shard_partitions
        return torch.Size(sharded_shape)

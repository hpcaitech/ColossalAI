from typing import Dict, List

from ..utils import merge_same_dim_mesh_list
from .misc import ShardingOutOfIndexError

__all__ = ["DimSpec", "ShardingException", "ShardingSpec"]

ALLGATHER_COST = 20
SHARD_COST = 5
STEP_PENALTY = 6
NAN = "nan"


class DimSpec:
    """
    Sharding spec for single dimension of the sharded tensor describe the sharding dimension of
    logical device mesh and give a method to compute the difference between them.
    This class is used internally in ShardingSpec.

    Argument:
        shard_list(List[int]): if shard_list is None, the dim spec will be 'R' type.
            Otherwise, the element in shard_list means the data will be sharded in that dimension.
    """

    _DIFFERENCE_DICT = None

    def __init__(self, shard_list):
        self.is_replica = len(shard_list) == 0
        self.shard_list = shard_list

    def __eq__(self, other):
        return str(self) == str(other)

    def __repr__(self):
        if self.is_replica:
            return "R"
        target = "S"
        for dim in self.shard_list:
            target += str(dim)
        return target

    @property
    def difference_dict(self):
        """
        Returns the difference dict, and lazily initializes it when needed

        Return:
            difference_dict(Dict[Tuple[int, int], Union[int, float, str]]):
                difference dict
        """
        if self._DIFFERENCE_DICT is None:
            self._DIFFERENCE_DICT = self._build_difference_2d_dict()

        return self._DIFFERENCE_DICT

    def dim_diff(self, other):
        """
        The difference between two DimSpec.

        Argument:
            other(DimSpec): the dim spec to compare with.

        Return:
            difference(int): the difference between two DimSpec.

        Example:
            dim_spec = DimSpec([0])
            other_dim_spec = DimSpec([0, 1])
            print(dim_spec.dim_diff(other_dim_spec))

        Output:
            5
        """
        difference = self.difference_dict[(str(self), str(other))]
        return difference

    @classmethod
    def _build_difference_2d_dict(cls):
        """
        Build a difference mapping for 2D device mesh case. It will be used to
        compute the difference between DimSpec pairs.
        """

        source_spec_list = ["R", "S0", "S1", "S01"]
        target_spec_list = ["R", "S0", "S1", "S01"]
        difference_dict = {}
        for source_spec in source_spec_list:
            for target_spec in target_spec_list:
                source_shard_list = cls._convert_str_to_shard_list(source_spec)
                target_shard_list = cls._convert_str_to_shard_list(target_spec)

                # source same as target
                if source_shard_list == target_shard_list:
                    difference = 0

                # all_gather(source) -> target
                elif (
                    len(source_shard_list) == len(target_shard_list) + 1 and source_shard_list[:-1] == target_shard_list
                ):
                    difference = ALLGATHER_COST

                # shard(source) -> target
                elif (
                    len(source_shard_list) == len(target_shard_list) - 1
                    and source_shard_list == target_shard_list[:-1]
                    and target_shard_list[-1] not in source_shard_list
                ):
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
                difference_dict[(source_spec, target_spec)] = difference

        return difference_dict

    @staticmethod
    def _convert_str_to_shard_list(str_spec):
        """
        Convert str_spec into shard_list.

        Argument:
            str_spec(str): dim spec in str type.
        """

        if str_spec == "R":
            return []
        if str_spec == "S0":
            return [0]
        if str_spec == "S1":
            return [1]
        if str_spec == "S01":
            return [0, 1]


class ShardingSpec:
    """
    Sharding spec describes how to shard a tensor with dim_size dimensions. For example for a 3D tensor, the sharding sequence
    [R, S0, S1] means not sharding the first dim, sharding the 3rd along the 1st device mesh axis (Process group)
    and sharding the 3th dim along the 2nd device mesh axis. Useful for say, 2D Tensor Parallel.

    Argument:
        dim_partition_dict(Dict[int, List[int]], optional): The key is the dimension of tensor to be sharded,
            and the value of the key describe which logical axis will be sharded in that dimension.
        sharding_sequence(List[DimSpec], optional): A straight view of ShardingSpec looks like [R, R, S0, S1].
    """

    def __init__(
        self, dim_size: int, dim_partition_dict: Dict[int, List[int]] = None, sharding_sequence: List[DimSpec] = None
    ):
        self.dims = dim_size
        self.dim_partition_dict = dim_partition_dict
        self.sharding_sequence = sharding_sequence
        if self.sharding_sequence is None:
            assert (
                self.dim_partition_dict is not None
            ), f"dim_partition_dict should not be None, if sharding_sequence is NoneType object."
            self.dim_partition_dict = merge_same_dim_mesh_list(
                dim_size=self.dims, dim_partition_dict=self.dim_partition_dict
            )
            self.sharding_sequence = self.convert_dict_to_shard_sequence()

        elif self.dim_partition_dict is None:
            assert (
                self.sharding_sequence is not None
            ), f"sharding_sequence should not be None, if dim_partition_dict is NoneType object."
            self.dim_partition_dict = self.convert_shard_sequence_to_dict()

        self._sanity_check()

    def _sanity_check(self):
        if len(self.sharding_sequence) > self.dims:
            raise ShardingOutOfIndexError(
                f"sharding_sequence should have {self.dims} elements, but got index {len(self.sharding_sequence)}."
            )

        if list(self.dim_partition_dict.keys()) and max(list(self.dim_partition_dict.keys())) >= self.dims:
            raise ShardingOutOfIndexError(
                f"the key of dim_partition_dict should be less than {self.dims}, but got {max(list(self.dim_partition_dict.keys()))}."
            )

    def __repr__(self):
        res_list = ["ShardingSpec:"]
        res_list.append(f"\n\tshard_sequence: " + ",".join(str(dimspec) for dimspec in self.sharding_sequence))
        return " ".join(res_list)

    def convert_dict_to_shard_sequence(self):
        """
        Convert dim_partition_dict into list of DimSpec, and assign it to sharding_sequence.
        """
        sharding_sequence = [DimSpec([])] * self.dims
        for dim, shard_list in self.dim_partition_dict.items():
            sharding_sequence[dim] = DimSpec(shard_list)
        return sharding_sequence

    def convert_shard_sequence_to_dict(self):
        """
        Convert sharding_sequence into dim_partition_dict.
        """
        new_dim_partition_dict = {}
        for index, dim_spec in enumerate(self.sharding_sequence):
            if not dim_spec.is_replica:
                if index not in new_dim_partition_dict:
                    new_dim_partition_dict[index] = []
                new_dim_partition_dict[index].extend(dim_spec.shard_list)
        return new_dim_partition_dict

    def spec_diff(self, other):
        """
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
        """
        assert len(self.sharding_sequence) == len(
            other.sharding_sequence
        ), f"Cannot compare difference for two sharding specs with different length."
        difference = 0
        for orig_dim_spec, other_dim_spec in zip(self.sharding_sequence, other.sharding_sequence):
            difference += orig_dim_spec.dim_diff(other_dim_spec)
        return difference

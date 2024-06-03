import operator
from functools import reduce

from colossalai.tensor.d_tensor.sharding_spec import ALLGATHER_COST, SHARD_COST, STEP_PENALTY, ShardingSpec


def test_dtensor_sharding_spec():
    dims = 4
    dim_partition_dict_0 = {0: [0, 1]}
    # DistSpec:
    #     shard_sequence: S01,R,R,R
    sharding_spec_0 = ShardingSpec(dims, dim_partition_dict=dim_partition_dict_0)
    assert str(sharding_spec_0.sharding_sequence) == "[S01, R, R, R]"

    dim_partition_dict_1 = {1: [0, 1]}
    # DistSpec:
    #     shard_sequence: R,S01,R,R
    sharding_spec_1 = ShardingSpec(dims, dim_partition_dict=dim_partition_dict_1)
    assert str(sharding_spec_1.sharding_sequence) == "[R, S01, R, R]"

    dim_spec_list_0 = [dim_spec for dim_spec in sharding_spec_0.sharding_sequence]
    dim_spec_list_1 = [dim_spec for dim_spec in sharding_spec_1.sharding_sequence]

    assert dim_spec_list_0[0].dim_diff(dim_spec_list_1[0]) == ALLGATHER_COST + STEP_PENALTY + ALLGATHER_COST
    assert dim_spec_list_0[1].dim_diff(dim_spec_list_1[1]) == SHARD_COST + STEP_PENALTY + SHARD_COST
    assert dim_spec_list_0[2].dim_diff(dim_spec_list_1[2]) == 0
    assert dim_spec_list_0[3].dim_diff(dim_spec_list_1[3]) == 0

    assert sharding_spec_0.spec_diff(sharding_spec_1) == reduce(
        operator.add, [dim_spec_list_0[i].dim_diff(dim_spec_list_1[i]) for i in range(dims)], 0
    )


if __name__ == "__main__":
    test_dtensor_sharding_spec()

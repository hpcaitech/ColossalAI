import pytest
import torch.distributed as dist
import torch.nn as nn

import colossalai
from colossalai.context.moe_context import MOE_CONTEXT
from colossalai.nn.layer.moe import EPExperts
from colossalai.testing import assert_equal_in_group, rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device
from colossalai.utils.moe import sync_moe_model_param

HIDDEN_SIZE = 4
INTERMEDIATE_SIZE = 8


def run_test(rank, world_size, port):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    expert_args = dict(hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE)

    MOE_CONTEXT.setup(42)    # MOE environment initialization
    exp0 = EPExperts(1, **expert_args)
    exp1 = EPExperts(2, **expert_args)
    exp2 = EPExperts(4, **expert_args)
    exp3 = EPExperts(8, **expert_args)

    assert exp0.num_local_experts == 1
    assert exp1.num_local_experts == 1
    assert exp2.num_local_experts == 1
    assert exp3.num_local_experts == 2
    # experts deployment passed

    parallel_info_dict = MOE_CONTEXT.parallel_info_dict
    rank = dist.get_rank()

    # group creation assert
    assert len(parallel_info_dict) == 3
    assert dist.get_rank(parallel_info_dict[4].ep_group) == rank
    assert dist.get_rank(parallel_info_dict[2].ep_group) == rank % 2
    assert dist.get_rank(parallel_info_dict[1].ep_group) == 0

    assert dist.get_rank(parallel_info_dict[4].dp_group) == 0
    assert dist.get_rank(parallel_info_dict[2].dp_group) == rank // 2
    assert dist.get_rank(parallel_info_dict[1].dp_group) == rank

    model = nn.ModuleList([exp0, exp1, exp2, exp3])
    model = model.to(get_current_device())
    sync_moe_model_param(model)

    # MOE experts layout success when ep_size = 1
    assert_equal_in_group(exp0.w1.data, parallel_info_dict[1].dp_group)
    assert_equal_in_group(exp0.b1.data, parallel_info_dict[1].dp_group)

    # MOE experts layout success when ep_size = 2
    assert_equal_in_group(exp1.w1.data, parallel_info_dict[2].dp_group)
    assert_equal_in_group(exp1.b1.data, parallel_info_dict[2].dp_group)


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_moe_initialization():
    spawn(run_test, 4)


if __name__ == '__main__':
    test_moe_initialization()

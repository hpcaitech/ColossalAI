import pytest
import torch.distributed as dist
import torch.nn as nn

import colossalai
from colossalai.context.moe_context import MOE_CONTEXT
from colossalai.nn.layer.moe import Experts
from colossalai.testing import assert_equal_in_group, rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device
from colossalai.utils.moe import sync_moe_model_param

D_MODEL = 4
D_FF = 8
CONFIG = dict()


def run_test(rank, world_size, port):
    world_size = 4
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    expert_module = nn.Linear
    expert_factor = dict(in_features=D_MODEL, out_features=D_FF, device=get_current_device())

    MOE_CONTEXT.setup(42)    # MOE environment initialization
    exp0 = Experts(expert_module, 1, **expert_factor)
    exp1 = Experts(expert_module, 2, **expert_factor)
    exp2 = Experts(expert_module, 4, **expert_factor)
    exp3 = Experts(expert_module, 8, **expert_factor)

    assert exp0.num_local_experts == 1
    assert exp1.num_local_experts == 1
    assert exp2.num_local_experts == 1
    assert exp3.num_local_experts == 2
    # experts deployment passed

    parallel_info_dict = MOE_CONTEXT.parallel_info_dict
    rank = dist.get_rank()

    assert len(parallel_info_dict) == 3
    assert dist.get_rank(parallel_info_dict[4].ep_group) == rank
    assert dist.get_rank(parallel_info_dict[2].ep_group) == rank % 2
    assert dist.get_rank(parallel_info_dict[1].ep_group) == 0

    assert dist.get_rank(parallel_info_dict[4].dp_group) == 0
    assert dist.get_rank(parallel_info_dict[2].dp_group) == rank // 2
    assert dist.get_rank(parallel_info_dict[1].dp_group) == rank
    # group creation passed

    model = nn.ModuleList([exp0, exp1, exp2, exp3])
    model = model.to(get_current_device())
    sync_moe_model_param(model)

    assert_equal_in_group(exp0.experts[0].weight.data, parallel_info_dict[1].dp_group)
    assert_equal_in_group(exp0.experts[0].bias.data, parallel_info_dict[1].dp_group)
    # MOE experts layout success when ep_size = 1

    assert_equal_in_group(exp1.experts[0].weight.data, parallel_info_dict[2].dp_group)
    assert_equal_in_group(exp1.experts[0].bias.data, parallel_info_dict[2].dp_group)
    # MOE experts layout success when ep_size = 2


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_moe_initialization():
    spawn(run_test, 4)


if __name__ == '__main__':
    test_moe_initialization()

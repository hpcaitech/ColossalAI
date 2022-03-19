from functools import partial
import pytest
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import colossalai
from colossalai.utils import free_port, get_current_device
from colossalai.nn.layer.moe import Experts
from colossalai.core import MOE_CONTEXT
from colossalai.utils.moe import sync_moe_model_param
from colossalai.testing import assert_equal_in_group

D_MODEL = 4
D_FF = 8
CONFIG = dict()


def run_test(rank, world_size, port):
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

    dist_dict = MOE_CONTEXT.information
    rank = dist.get_rank()

    assert len(dist_dict) == 3
    assert dist.get_rank(dist_dict[4].ep_group) == rank
    assert dist.get_rank(dist_dict[2].ep_group) == rank % 2
    assert dist.get_rank(dist_dict[1].ep_group) == 0

    assert dist.get_rank(dist_dict[4].dp_group) == 0
    assert dist.get_rank(dist_dict[2].dp_group) == rank // 2
    assert dist.get_rank(dist_dict[1].dp_group) == rank
    # group creation passed

    model = nn.ModuleList([exp0, exp1, exp2, exp3])
    model = model.to(get_current_device())
    sync_moe_model_param(model)

    assert_equal_in_group(exp0.experts[0].weight.data, dist_dict[1].dp_group)
    assert_equal_in_group(exp0.experts[0].bias.data, dist_dict[1].dp_group)
    # MOE experts layout success when ep_size = 1

    assert_equal_in_group(exp1.experts[0].weight.data, dist_dict[2].dp_group)
    assert_equal_in_group(exp1.experts[0].bias.data, dist_dict[2].dp_group)
    # MOE experts layout success when ep_size = 2


@pytest.mark.dist
def test_moe_initialization():
    world_size = 4
    run_func = partial(run_test, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_moe_initialization()

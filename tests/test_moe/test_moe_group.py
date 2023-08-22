import pytest
import torch.distributed as dist
import torch.nn as nn

import colossalai
from colossalai.context.moe_context import MOE_CONTEXT
from colossalai.nn.layer.moe import EPMLPExperts, TPMLPExperts
from colossalai.testing import assert_equal_in_group, rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device
from colossalai.utils.moe import sync_moe_model_param

HIDDEN_SIZE = 4
INTERMEDIATE_SIZE = 8


def run_moe_init(expert_cls):
    expert_args = dict(hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE)
    exp0 = expert_cls(1, **expert_args)
    exp1 = expert_cls(2, **expert_args)
    exp2 = expert_cls(4, **expert_args)
    exp3 = expert_cls(8, **expert_args)

    if expert_cls is EPMLPExperts:
        assert exp0.num_local_experts == 1
        assert exp1.num_local_experts == 1
        assert exp2.num_local_experts == 1
        assert exp3.num_local_experts == 2
    else:
        assert exp0.num_local_experts == 1
        assert exp1.num_local_experts == 2
        assert exp2.num_local_experts == 4
        assert exp3.num_local_experts == 8

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
    assert_equal_in_group(exp0.wi.data, parallel_info_dict[1].dp_group)
    assert_equal_in_group(exp0.wo.data, parallel_info_dict[1].dp_group)

    # MOE experts layout success when ep_size = 2
    assert_equal_in_group(exp1.wi.data, parallel_info_dict[2].dp_group)
    assert_equal_in_group(exp1.wo.data, parallel_info_dict[2].dp_group)


def _run_test(rank, world_size, port, expert_cls):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    MOE_CONTEXT.setup(seed=42)
    run_moe_init(expert_cls)


@pytest.mark.dist
@pytest.mark.parametrize("expert_cls", [EPMLPExperts, TPMLPExperts])
@rerun_if_address_is_in_use()
def test_moe_initialization(expert_cls):
    spawn(_run_test, 4, expert_cls=expert_cls)


if __name__ == '__main__':
    test_moe_initialization(EPMLPExperts)
    test_moe_initialization(TPMLPExperts)

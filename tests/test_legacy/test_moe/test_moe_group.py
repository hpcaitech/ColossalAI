import pytest
import torch.distributed as dist
import torch.nn as nn

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.legacy.moe.manager import MOE_MANAGER
from colossalai.legacy.moe.utils import sync_moe_model_param

# from colossalai.shardformer.layer.moe import MLPExperts
from colossalai.testing import assert_equal_in_group, rerun_if_address_is_in_use, spawn

HIDDEN_SIZE = 4
INTERMEDIATE_SIZE = 8


def run_moe_init(expert_parallel):
    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(parallel=expert_parallel)
    expert_args = dict(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        expert_parallel=expert_parallel,
    )
    exp0 = MLPExperts(1, **expert_args)
    exp1 = MLPExperts(2, **expert_args)
    exp2 = MLPExperts(4, **expert_args)

    if expert_parallel == "EP":
        assert exp0.num_local_experts == 1
        assert exp1.num_local_experts == 1
        assert exp2.num_local_experts == 2
    else:
        assert exp0.num_local_experts == 1
        assert exp1.num_local_experts == 2
        assert exp2.num_local_experts == 4

    parallel_info_dict = MOE_MANAGER.parallel_info_dict
    rank = dist.get_rank()

    # group creation assert
    assert len(parallel_info_dict) == 2
    assert dist.get_rank(parallel_info_dict[2].ep_group) == rank % 2
    assert dist.get_rank(parallel_info_dict[1].ep_group) == 0

    assert dist.get_rank(parallel_info_dict[2].dp_group) == rank // 2
    assert dist.get_rank(parallel_info_dict[1].dp_group) == rank

    model = nn.ModuleList([exp0, exp1, exp2])
    model = model.to(get_accelerator().get_current_device())
    sync_moe_model_param(model)

    # MOE experts layout success when ep_size = 1
    assert_equal_in_group(exp0.wi.data, parallel_info_dict[1].dp_group)
    assert_equal_in_group(exp0.wo.data, parallel_info_dict[1].dp_group)

    # MOE experts layout success when ep_size = 2
    assert_equal_in_group(exp1.wi.data, parallel_info_dict[2].dp_group)
    assert_equal_in_group(exp1.wo.data, parallel_info_dict[2].dp_group)


def _run_test(rank, world_size, port, expert_parallel):
    colossalai.launch(
        rank=rank,
        world_size=world_size,
        host="localhost",
        port=port,
        backend="nccl",
    )
    run_moe_init(expert_parallel)


@pytest.mark.skip(reason="moe need to be refactored")
@pytest.mark.dist
@pytest.mark.parametrize("expert_parallel", ["EP", "TP"])
@rerun_if_address_is_in_use()
def test_moe_initialization(expert_parallel):
    spawn(_run_test, 2, expert_parallel=expert_parallel)


if __name__ == "__main__":
    test_moe_initialization("EP")
    test_moe_initialization("TP")

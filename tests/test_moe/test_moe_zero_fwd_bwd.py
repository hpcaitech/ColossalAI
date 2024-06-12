import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import colossalai
from colossalai.moe.manager import MOE_MANAGER
from colossalai.tensor.moe_tensor.api import is_moe_tensor
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all
from colossalai.zero.low_level.low_level_optim import LowLevelZeroOptimizer
from colossalai.zero.low_level.low_level_strategy import LowLevelOptStrategy, MoeZeroStrategy
from tests.test_moe.moe_utils import MoeModel, delete_moe_info, loose_close, sync_local_from_ep


def run_zero_test(local_rank):
    dp_size = world_size = dist.get_world_size()
    assert world_size >= 4, f"{world_size=}: at least 4 processes are required for this test (ep=2, moe_dp=2)"
    criterion = torch.nn.CrossEntropyLoss()

    ep_size = 2
    extra_dp_size = world_size // ep_size

    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(parallel="EP", mode="fixed", fixed_dp_size=extra_dp_size, fixed_ep_size=ep_size, fixed_pp_size=1)

    zero_model = MoeModel().bfloat16().cuda()

    dp_group = dist.group.WORLD
    ep_group = MOE_MANAGER.parallel_info_dict[ep_size].ep_group
    moe_extra_dp_group = MOE_MANAGER.parallel_info_dict[ep_size].dp_group

    zero_params = list(filter(lambda x: not is_moe_tensor(x), zero_model.parameters()))
    moe_params = list(filter(lambda x: is_moe_tensor(x), zero_model.parameters()))
    print(f"{len(zero_params)=}, {len(moe_params)=}")
    lr = 1e-3
    zero_optimizer = torch.optim.SGD(zero_model.parameters(), lr=lr)
    zero_optimizer.param_groups.clear()
    zero_optimizer.add_param_group({"params": zero_params})
    zero_optimizer.add_param_group({"params": moe_params})

    strategies = [
        LowLevelOptStrategy(
            param_group=zero_optimizer.param_groups[0],
            process_group=dp_group,
            overlap_communication=False,
            partition_grad=True,
        ),
        MoeZeroStrategy(
            param_group=zero_optimizer.param_groups[1],
            process_group=moe_extra_dp_group,
            overlap_communication=True,
            partition_grad=False,
        ),
    ]
    zero_optimizer = LowLevelZeroOptimizer(
        zero_optimizer,
        strategies,
    )

    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(parallel=None)
    ddp_model = DDP(MoeModel().bfloat16().cuda(), static_graph=True)
    delete_moe_info(ddp_model)
    torch_optim = torch.optim.SGD(ddp_model.parameters(), lr=lr)
    sync_local_from_ep(ddp_model, zero_model)

    seed_all(42 + local_rank)
    data = torch.randn(16, 4).bfloat16().cuda()
    label = torch.randint(0, 4, (16,)).cuda()

    ddp_model.train()
    zero_model.train()
    ddp_out = criterion(ddp_model(data), label).float()
    zero_out = criterion(zero_model(data), label).float()
    assert torch.allclose(ddp_out, zero_out)
    print(f"{local_rank=} {ddp_out.mean()=}")

    ddp_out.backward()
    zero_optimizer.backward(zero_out)

    for (zero_name, zero_param), (ddp_name, ddp_param) in zip(
        zero_model.named_parameters(), ddp_model.named_parameters()
    ):
        torch_grad = ddp_param.grad
        zero_grad = zero_optimizer.get_param_grad(zero_param)
        if is_moe_tensor(zero_param):
            moe_grad_list = [torch.empty_like(zero_grad) for _ in range(ep_size)]
            dist.all_gather(moe_grad_list, zero_grad, group=ep_group)
            zero_grad = torch.cat(moe_grad_list, dim=0)
        loose_close(torch_grad, zero_grad, dtype=torch_grad.dtype)


def run_dist(rank, world_size, port, stage):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_zero_test(rank, stage=stage)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4])
@rerun_if_address_is_in_use()
def test_moe_zero_model(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_moe_zero_model(world_size=4)

import pytest
import torch

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.moe.manager import MOE_MANAGER
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all
from tests.test_moe.moe_utils import MoeModel, delete_moe_info, run_fwd_bwd, sync_local_from_ep


def run_zero_test(local_rank, stage=1):
    criterion = torch.nn.CrossEntropyLoss()

    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(parallel="EP")
    moe_model = MoeModel().bfloat16()
    moe_optimizer = torch.optim.Adam(moe_model.parameters())
    moe_plugin = LowLevelZeroPlugin(stage=stage, precision="bf16")
    moe_booster = Booster(plugin=moe_plugin)
    moe_model, moe_optimizer, _, _, _ = moe_booster.boost(moe_model, moe_optimizer)

    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(parallel=None)
    zero_model = MoeModel().bfloat16()
    delete_moe_info(zero_model)
    zero_optimizer = torch.optim.Adam(zero_model.parameters())
    zero_plugin = LowLevelZeroPlugin(stage=stage, precision="bf16")
    zero_booster = Booster(plugin=zero_plugin)
    zero_model, zero_optimizer, _, _, _ = zero_booster.boost(zero_model, zero_optimizer)
    sync_local_from_ep(zero_model, moe_model)

    data = torch.randn(16, 4).bfloat16().cuda()
    label = torch.randint(0, 4, (16,)).cuda()

    zero_out = run_fwd_bwd(zero_model, data, label, criterion, zero_optimizer)
    moe_out = run_fwd_bwd(moe_model, data, label, criterion, moe_optimizer)
    assert torch.allclose(zero_out, moe_out)

    for (moe_name, moe_param), (zero_name, zero_param) in zip(
        moe_model.module.named_parameters(), zero_model.module.named_parameters()
    ):
        assert moe_name == zero_name
        moe_grad_list = moe_optimizer._grad_store.get_partitioned_gradients_by_param_id(0, id(moe_param))
        zero_grad_list = zero_optimizer._grad_store.get_partitioned_gradients_by_param_id(0, id(zero_param))
        if hasattr(moe_param, "moe_info"):
            assert len(moe_grad_list) == 0
            if stage == 1:
                zero_grad = zero_grad_list[local_rank].view(moe_param.grad.shape)
            else:
                zero_grad = zero_grad_list[0].view(moe_param.grad.shape)
            assert torch.allclose(
                moe_param.grad, zero_grad, atol=1e-5
            ), f"zero grad:\n{moe_param.grad}\ntorch grad:\n{zero_grad}\nmax diff: {(moe_param.grad - zero_grad).abs().max()}, mean diff: {(moe_param.grad - zero_grad).abs().mean()}"
        else:
            assert len(moe_grad_list) > 0
            assert len(moe_grad_list) == len(zero_grad_list)
            for moe_grad, zero_grad in zip(moe_grad_list, zero_grad_list):
                assert torch.allclose(moe_grad, zero_grad)


def run_dist(rank, world_size, port, stage):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    seed_all(42 + rank)
    run_zero_test(rank, stage=stage)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2])
@pytest.mark.parametrize("stage", [1, 2])
@rerun_if_address_is_in_use()
def test_moe_zero_model(world_size, stage):
    spawn(run_dist, world_size, stage=stage)


if __name__ == "__main__":
    test_moe_zero_model(world_size=2, stage=1)

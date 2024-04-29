import pytest
import torch

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.moe.manager import MOE_MANAGER
from colossalai.tensor.moe_tensor.api import is_moe_tensor
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all
from tests.test_moe.moe_utils import MoeModel, delete_moe_info, loose_close, run_fwd_bwd, sync_local_from_ep


def run_zero_test(local_rank, stage=1):
    criterion = torch.nn.CrossEntropyLoss()

    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(parallel="EP")
    moe_model = MoeModel().bfloat16()
    moe_optimizer = torch.optim.Adam(moe_model.parameters(), lr=1.0)
    moe_plugin = LowLevelZeroPlugin(stage=stage, precision="bf16")
    moe_booster = Booster(plugin=moe_plugin)
    moe_model, moe_optimizer, _, _, _ = moe_booster.boost(moe_model, moe_optimizer)

    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(parallel=None)
    zero_model = MoeModel().bfloat16()
    delete_moe_info(zero_model)
    sync_local_from_ep(zero_model, moe_model)
    zero_optimizer = torch.optim.Adam(zero_model.parameters(), lr=1.0)
    zero_plugin = LowLevelZeroPlugin(stage=stage, precision="bf16")
    zero_booster = Booster(plugin=zero_plugin)
    zero_model, zero_optimizer, _, _, _ = zero_booster.boost(zero_model, zero_optimizer)

    for (moe_name, moe_param), (zero_name, zero_param) in zip(
        moe_model.named_parameters(), zero_model.named_parameters()
    ):
        if ".experts." in moe_name:
            continue
        assert moe_name == zero_name
        assert torch.allclose(
            moe_param.data, zero_param.data
        ), f"{moe_name}\ntorch_param {moe_param.data}\nzero_param {zero_param.data}"

    for _ in range(1):
        data = torch.randn(2, 4).bfloat16().cuda()
        label = torch.randint(0, 4, (2,)).cuda()

        moe_out = run_fwd_bwd(moe_model, data, label, criterion, moe_optimizer)
        zero_out = run_fwd_bwd(zero_model, data, label, criterion, zero_optimizer)
        assert torch.allclose(zero_out, moe_out)
        moe_optimizer.step()
        zero_optimizer.step()

        for (moe_name, moe_param), (zero_name, zero_param) in zip(
            moe_model.named_parameters(), zero_model.named_parameters()
        ):
            assert moe_name == zero_name
            if is_moe_tensor(moe_param):
                param_size = moe_param.shape[0]
                zero_param = zero_param[local_rank * param_size : (local_rank + 1) * param_size]
            loose_close(moe_param.data, zero_param.data, dtype=moe_param.dtype)

        moe_optimizer.zero_grad()
        zero_optimizer.zero_grad()


def run_dist(rank, world_size, port, stage):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    seed_all(42 + rank)
    run_zero_test(rank, stage=stage)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2])
@pytest.mark.parametrize("stage", [1, 2])
@rerun_if_address_is_in_use()
def test_moe_zero_optim(world_size, stage):
    spawn(run_dist, world_size, stage=stage)


if __name__ == "__main__":
    test_moe_zero_optim(world_size=2, stage=1)

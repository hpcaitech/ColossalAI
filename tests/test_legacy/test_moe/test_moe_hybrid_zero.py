import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.booster.plugin.low_level_zero_plugin import LowLevelZeroModel
from colossalai.legacy.moe.manager import MOE_MANAGER
from colossalai.tensor.moe_tensor.api import is_moe_tensor
from colossalai.testing import rerun_if_address_is_in_use, spawn
from tests.test_moe.moe_utils import MoeModel


def run_fwd_bwd(model, data, label, criterion, optimizer, enable_autocast=False):
    model.train()
    with torch.cuda.amp.autocast(enabled=enable_autocast):
        if criterion:
            y = model(data)
            loss = criterion(y, label)
        else:
            loss = model(data, label)
        loss = loss.float()

    if isinstance(model, LowLevelZeroModel):
        optimizer.backward(loss / 2)
    else:
        loss.backward()
    return y


def run_zero_optim_test(local_rank, world_size, stage=1):
    criterion = torch.nn.CrossEntropyLoss()
    data = torch.randn(16, 4).cuda()
    label = torch.randint(0, 4, (16,)).cuda()

    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(parallel=None)
    torch_model = MoeModel()
    torch_optimizer = torch.optim.Adam(torch_model.parameters())
    torch_model = torch_model.cuda()

    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(max_ep_size=2, use_ep_inside=False, parallel="EP")
    zero_model = MoeModel()
    extra_dp_group = MOE_MANAGER.parallel_info_dict[2].dp_group
    ep_rank = dist.get_rank(MOE_MANAGER.parallel_info_dict[2].ep_group)
    ep_size = MOE_MANAGER.parallel_info_dict[2].ep_size
    for zero_param, torch_param in zip(zero_model.parameters(), torch_model.parameters()):
        if is_moe_tensor(zero_param):
            num_expert = torch_param.data.shape[0]
            zero_param.data.copy_(
                torch_param.data[ep_rank * (num_expert // ep_size) : (ep_rank + 1) * (num_expert // ep_size)]
                .detach()
                .clone()
            )
        else:
            zero_param.data.copy_(torch_param.data.detach().clone())
    zero_optimizer = torch.optim.Adam(zero_model.parameters())
    plugin = LowLevelZeroPlugin(stage=stage, precision="fp32")
    plugin.zero_optim_kwargs["moe_extra_dp_process_group"] = extra_dp_group
    booster = Booster(plugin=plugin)
    zero_model, zero_optimizer, _, _, _ = booster.boost(zero_model, zero_optimizer)

    run_fwd_bwd(torch_model, data, label, criterion, None)
    torch_optimizer.step()
    run_fwd_bwd(zero_model, data, label, criterion, zero_optimizer)
    zero_optimizer.step()

    for (torch_name, torch_param), (zero_name, zero_param) in zip(
        torch_model.named_parameters(), zero_model.named_parameters()
    ):
        if is_moe_tensor(zero_param):
            num_expert = torch_param.data.shape[0]
            torch_param.data = torch_param.data[
                ep_rank * (num_expert // ep_size) : (ep_rank + 1) * (num_expert // ep_size)
            ]
        assert torch.allclose(
            torch_param.data, zero_param.data, atol=1e-4
        ), f"{torch_name}\ntorch_param {torch_param.data}\nzero_param {zero_param.data}"


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_zero_optim_test(rank, world_size, stage=1)
    run_zero_optim_test(rank, world_size, stage=2)


@pytest.mark.skip(reason="moe need to be refactored")
@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4])
@rerun_if_address_is_in_use()
def test_moe_zero_optim(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_moe_zero_optim(world_size=4)

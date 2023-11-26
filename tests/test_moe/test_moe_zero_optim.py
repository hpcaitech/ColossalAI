import pytest
import torch

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.booster.plugin.low_level_zero_plugin import LowLevelZeroModel
from colossalai.moe.manager import MOE_MANAGER
from colossalai.testing import rerun_if_address_is_in_use, spawn
from tests.test_moe.moe_utils import MoeGradientHandler, MoeModel


def split_ddp_grad(grad, world_size):
    with torch.no_grad():
        grad = grad.clone().detach().flatten()
        padding_size = (world_size - grad.numel() % world_size) % world_size
        if padding_size > 0:
            grad = torch.nn.functional.pad(grad, [0, padding_size])
        splited_grad = grad.split(grad.numel() // world_size)
    return splited_grad


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
        optimizer.backward(loss)
    else:
        loss.backward()
    return y


def run_zero_optim_test(local_rank, world_size, stage=1):
    criterion = torch.nn.CrossEntropyLoss()

    zero_model = MoeModel()
    zero_optimizer = torch.optim.Adam(zero_model.parameters())
    plugin = LowLevelZeroPlugin(stage=stage, precision="fp32")
    booster = Booster(plugin=plugin)
    zero_model, zero_optimizer, _, _, _ = booster.boost(zero_model, zero_optimizer)

    torch_model = MoeModel()
    for zero_param, torch_param in zip(zero_model.parameters(), torch_model.parameters()):
        torch_param.data.copy_(zero_param.data)
    torch_optimizer = torch.optim.Adam(torch_model.parameters())
    torch_model = torch_model.cuda()
    grad_handler = MoeGradientHandler(torch_model)

    for _ in range(2):
        data = torch.randn(16, 4).cuda() / (local_rank + 1)
        label = torch.randint(0, 4, (16,)).cuda()
        run_fwd_bwd(torch_model, data, label, criterion, None)
        run_fwd_bwd(zero_model, data, label, criterion, zero_optimizer)
        grad_handler.handle_gradient()

        torch_optimizer.step()
        zero_optimizer.step()

        for (torch_name, torch_param), (zero_name, zero_param) in zip(
            torch_model.named_parameters(), zero_model.named_parameters()
        ):
            assert torch.allclose(
                torch_param.data, zero_param.data
            ), f"{torch_name}\ntorch_param {torch_param.data}\nzero_param {zero_param.data}"

        torch_optimizer.zero_grad()
        zero_optimizer.zero_grad()


def run_dist(rank, world_size, port):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    MOE_MANAGER.setup(parallel="EP")
    run_zero_optim_test(rank, world_size, stage=1)
    run_zero_optim_test(rank, world_size, stage=2)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2])
@rerun_if_address_is_in_use()
def test_moe_zero_optim(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_moe_zero_optim(world_size=2)

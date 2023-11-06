import pytest
import torch

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.booster.plugin.low_level_zero_plugin import LowLevelZeroModel
from colossalai.moe.manager import MOE_MANAGER
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all
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


def run_zero_test(local_rank, world_size, stage=1):
    criterion = torch.nn.CrossEntropyLoss()

    zero_model = MoeModel()
    optimizer = torch.optim.Adam(zero_model.parameters())
    plugin = LowLevelZeroPlugin(stage=stage, precision="fp32")
    booster = Booster(plugin=plugin)
    zero_model, optimizer, _, _, _ = booster.boost(zero_model, optimizer)

    torch_model = MoeModel()
    for zero_param, torch_param in zip(zero_model.parameters(), torch_model.parameters()):
        torch_param.data.copy_(zero_param.data)
    torch_model = torch_model.cuda()
    grad_handler = MoeGradientHandler(torch_model)

    # assert zero model
    for (torch_name, torch_param), (zero_name, zero_param) in zip(
        torch_model.named_parameters(), zero_model.module.named_parameters()
    ):
        assert zero_name == torch_name
        assert torch.allclose(zero_param.data, torch_param.data)

    data = torch.randn(16, 4).cuda()
    label = torch.randint(0, 4, (16,)).cuda()

    torch_out = run_fwd_bwd(torch_model, data, label, criterion, None)
    zero_out = run_fwd_bwd(zero_model, data, label, criterion, optimizer)
    assert torch.allclose(torch_out, zero_out)
    grad_handler.handle_gradient()

    for (zero_name, zero_param), (torch_name, torch_param) in zip(
        zero_model.module.named_parameters(), torch_model.named_parameters()
    ):
        assert zero_name == torch_name
        zero_grad_list = optimizer._grad_store.get_partitioned_gradients_by_param_id(0, id(zero_param))
        if hasattr(zero_param, "moe_info"):
            assert len(zero_grad_list) == 0
            assert torch.allclose(zero_param.grad, torch_param.grad)
        else:
            assert len(zero_grad_list) > 0
            torch_grad_list = split_ddp_grad(torch_param.grad, world_size)
            if stage == 2:
                torch_grad_list = torch_grad_list[local_rank : local_rank + 1]
            assert len(zero_grad_list) == len(torch_grad_list)
            for zero_grad, torch_grad in zip(zero_grad_list, torch_grad_list):
                assert torch.allclose(zero_grad, torch_grad)


def run_dist(rank, world_size, port):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    MOE_MANAGER.setup(parallel="EP")
    seed_all(42 + rank)
    run_zero_test(rank, world_size, stage=1)
    run_zero_test(rank, world_size, stage=2)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2])
@rerun_if_address_is_in_use()
def test_moe_zero_model(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_moe_zero_model(world_size=2)

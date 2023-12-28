import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.booster.plugin.low_level_zero_plugin import LowLevelZeroModel
from colossalai.moe import SparseMLP
from colossalai.moe.manager import MOE_MANAGER
from colossalai.tensor.moe_tensor.api import get_ep_group, get_ep_size
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all
from tests.test_moe.moe_utils import MoeModel


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


def delete_moe_info(model):
    for name, param in model.named_parameters():
        if hasattr(param, "moe_info"):
            delattr(param, "moe_info")


def sync_local_from_ep(local_model: SparseMLP, ep_model: SparseMLP, assert_grad_flag: bool = False) -> None:
    """Sync the parameters of tp model from ep model

    Args:
        local_model (MoeModule)
        ep_model (MoeModule)
    """
    for (local_name, local_param), (ep_name, ep_param) in zip(
        local_model.named_parameters(), ep_model.named_parameters()
    ):
        assert local_name == ep_name, print(f"{local_name} != {ep_name}")
        if "experts" not in local_name:
            if assert_grad_flag:
                assert torch.allclose(local_param, ep_param), f"local_param: {local_param}, ep_param: {ep_param}"
                assert torch.allclose(local_param.grad, ep_param.grad)
            else:
                local_param.data.copy_(ep_param.data)
            continue

        # gather param from ep model
        param_list = [torch.zeros_like(ep_param) for _ in range(get_ep_size(ep_param))]
        dist.all_gather(param_list, ep_param, group=get_ep_group(ep_param))
        all_param = torch.cat(param_list, dim=0)
        if assert_grad_flag:
            grad_list = [torch.zeros_like(ep_param) for _ in range(get_ep_size(ep_param))]
            dist.all_gather(grad_list, ep_param.grad, group=get_ep_group(ep_param))
            all_grad = torch.cat(grad_list, dim=0)

        if assert_grad_flag:
            assert torch.allclose(local_param, all_param)
            assert torch.allclose(local_param.grad, all_grad)
        else:
            local_param.data.copy_(all_param.data)


def run_zero_test(local_rank, stage=1):
    criterion = torch.nn.CrossEntropyLoss()

    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(parallel="EP")
    zero_model = MoeModel().bfloat16()
    zero_optimizer = torch.optim.Adam(zero_model.parameters())
    zero_plugin = LowLevelZeroPlugin(stage=stage, precision="bf16")
    zero_booster = Booster(plugin=zero_plugin)
    zero_model, zero_optimizer, _, _, _ = zero_booster.boost(zero_model, zero_optimizer)

    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(parallel=None)
    torch_model = MoeModel().bfloat16()
    delete_moe_info(torch_model)
    torch_optimizer = torch.optim.Adam(torch_model.parameters())
    torch_plugin = LowLevelZeroPlugin(stage=stage, precision="bf16")
    torch_booster = Booster(plugin=torch_plugin)
    torch_model, torch_optimizer, _, _, _ = torch_booster.boost(torch_model, torch_optimizer)
    sync_local_from_ep(torch_model, zero_model)

    data = torch.randn(16, 4).bfloat16().cuda()
    label = torch.randint(0, 4, (16,)).cuda()

    torch_out = run_fwd_bwd(torch_model, data, label, criterion, torch_optimizer)
    zero_out = run_fwd_bwd(zero_model, data, label, criterion, zero_optimizer)
    assert torch.allclose(torch_out, zero_out)

    for (zero_name, zero_param), (torch_name, torch_param) in zip(
        zero_model.module.named_parameters(), torch_model.module.named_parameters()
    ):
        assert zero_name == torch_name
        zero_grad_list = zero_optimizer._grad_store.get_partitioned_gradients_by_param_id(0, id(zero_param))
        torch_grad_list = torch_optimizer._grad_store.get_partitioned_gradients_by_param_id(0, id(torch_param))
        if hasattr(zero_param, "moe_info"):
            assert len(zero_grad_list) == 0
            if stage == 1:
                torch_grad = torch_grad_list[local_rank].view(zero_param.grad.shape)
            else:
                torch_grad = torch_grad_list[0].view(zero_param.grad.shape)
            assert torch.allclose(
                zero_param.grad, torch_grad, atol=1e-5
            ), f"zero grad:\n{zero_param.grad}\ntorch grad:\n{torch_grad}\nmax diff: {(zero_param.grad - torch_grad).abs().max()}, mean diff: {(zero_param.grad - torch_grad).abs().mean()}"
        else:
            assert len(zero_grad_list) > 0
            assert len(zero_grad_list) == len(torch_grad_list)
            for zero_grad, torch_grad in zip(zero_grad_list, torch_grad_list):
                assert torch.allclose(zero_grad, torch_grad)


def run_dist(rank, world_size, port):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    seed_all(42 + rank)
    run_zero_test(rank, stage=1)
    run_zero_test(rank, stage=2)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2])
@rerun_if_address_is_in_use()
def test_moe_zero_model(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_moe_zero_model(world_size=2)

import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.booster.plugin.low_level_zero_plugin import LowLevelZeroModel
from colossalai.legacy.moe.manager import MOE_MANAGER

# from colossalai.shardformer.layer.moe import apply_load_balance
from colossalai.tensor.moe_tensor.api import is_moe_tensor
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

    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(
        parallel="EP",
    )
    zero_model = MoeModel(enable_load_balance=True)
    zero_optimizer = torch.optim.Adam(zero_model.parameters())
    plugin = LowLevelZeroPlugin(stage=stage, precision="bf16", verbose=True)
    booster = Booster(plugin=plugin)
    zero_model, zero_optimizer, _, _, _ = booster.boost(zero_model, zero_optimizer)

    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(parallel="EP")
    torch_model = MoeModel()
    for zero_param, torch_param in zip(zero_model.parameters(), torch_model.parameters()):
        torch_param.data.copy_(zero_param.data)
    torch_optimizer = torch.optim.Adam(torch_model.parameters())
    torch_model = torch_model.cuda().bfloat16()
    grad_handler = MoeGradientHandler(torch_model)

    # run to update expert load
    data = torch.randn(16, 4).cuda().bfloat16() / 1000 / (local_rank + 1)
    label = torch.randint(0, 4, (16,)).cuda()

    # run torch model twice
    run_fwd_bwd(torch_model, data, label, criterion, None)
    grad_handler.handle_gradient()
    torch_optimizer.step()
    torch_optimizer.zero_grad()
    run_fwd_bwd(torch_model, data, label, criterion, None)
    grad_handler.handle_gradient()

    # get optim and load status in zero model
    run_fwd_bwd(zero_model, data, label, criterion, zero_optimizer)
    zero_optimizer.step()
    zero_optimizer.zero_grad()
    with torch.no_grad():
        origin_out = zero_model(data)

    # load balance
    apply_load_balance(zero_model, zero_optimizer)

    # run again to test
    zero_out = run_fwd_bwd(zero_model, data, label, criterion, zero_optimizer)
    torch.allclose(origin_out, zero_out)

    # assert optim
    torch_optimizer.step()
    torch_out = run_fwd_bwd(torch_model, data, label, criterion, None)
    zero_optimizer.step()
    zero_out = run_fwd_bwd(zero_model, data, label, criterion, zero_optimizer)
    assert torch.allclose(zero_out, torch_out, atol=3e-5), f"zero_out:{zero_out}\ntorch_out{torch_out}"


def run_hybrid_zero_optim_test(local_rank, world_size, stage=1):
    criterion = torch.nn.CrossEntropyLoss()
    data = torch.randn(16, 4).cuda()
    label = torch.randint(0, 4, (16,)).cuda()

    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(parallel=None)
    torch_model = MoeModel()
    torch_optimizer = torch.optim.Adam(torch_model.parameters())
    torch_model = torch_model.cuda()

    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(
        max_ep_size=2,
        use_ep_inside=False,
        parallel="EP",
    )
    zero_model = MoeModel(enable_load_balance=True)
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

    # run torch for twice
    run_fwd_bwd(torch_model, data, label, criterion, None)
    torch_optimizer.step()
    torch_optimizer.zero_grad()
    run_fwd_bwd(torch_model, data, label, criterion, None)
    torch_optimizer.step()

    # run zero
    run_fwd_bwd(zero_model, data, label, criterion, zero_optimizer)
    zero_optimizer.step()
    zero_optimizer.zero_grad()
    with torch.no_grad():
        origin_out = zero_model(data)

    # load balance
    apply_load_balance(zero_model, zero_optimizer)

    # assert out
    zero_out = run_fwd_bwd(zero_model, data, label, criterion, zero_optimizer)
    torch.allclose(origin_out, zero_out)

    # assert optim
    zero_optimizer.step()
    zero_out = run_fwd_bwd(zero_model, data, label, criterion, zero_optimizer)
    torch_out = run_fwd_bwd(torch_model, data, label, criterion, None)
    # TODO: high atol, check if bug exists
    assert torch.allclose(zero_out, torch_out, atol=8e-4), f"zero_out:{zero_out}\ntorch_out{torch_out}"


def run_dist(rank, world_size, port):
    colossalai.launch(
        rank=rank,
        world_size=world_size,
        host="localhost",
        port=port,
        backend="nccl",
    )
    run_zero_optim_test(rank, world_size, stage=1)
    run_zero_optim_test(rank, world_size, stage=2)
    run_hybrid_zero_optim_test(rank, world_size, stage=1)
    run_hybrid_zero_optim_test(rank, world_size, stage=2)


@pytest.mark.skip(reason="moe need to be refactored")
@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4])
@rerun_if_address_is_in_use()
def test_moe_load_balance(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_moe_load_balance(world_size=4)

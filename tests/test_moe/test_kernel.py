import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.moe import SparseMLP
from colossalai.moe.manager import MOE_MANAGER
from colossalai.testing import rerun_if_address_is_in_use, spawn

BATCH_SIZE = 4
NUM_EXPERTS = 4


def check_equal(tensor_a, tensor_b, atol=1e-06):
    assert torch.allclose(tensor_a, tensor_b, rtol=0, atol=atol) is True


def run_routing(rank, world_size, port, rs=2, hidden_size=128, data_type=torch.float32, topk=1):
    # Here we do not need TF32, since it brings absolute error on results
    torch.backends.cuda.matmul.allow_tf32 = False

    colossalai.launch(config=dict(), rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    local_rank = dist.get_rank()

    MOE_MANAGER.setup(parallel="EP")  # MOE environment initialization
    MOE_MANAGER.reset_loss()
    torch.manual_seed(rs + local_rank)  # set each process has different random seed

    # get randomized data
    tokens = torch.randn(
        BATCH_SIZE, hidden_size, dtype=data_type, device=get_accelerator().get_current_device(), requires_grad=True
    )

    layer = SparseMLP(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_experts=NUM_EXPERTS,
        router_top_k=topk,
        router_capacity_factor_train=1.0,
    )
    layer = layer.to(get_accelerator().get_current_device())
    if data_type == torch.float16:
        layer = layer.half()

    # use matrix multiplication instead of COL_MOE_KERNEL in MOE dispatch and combine
    layer.enable_kernel = False
    old_out = layer(tokens)
    ech = old_out.shape
    grad = torch.randn(ech, device=get_accelerator().get_current_device())
    old_out.backward(grad)  # get gradient

    # save all results
    o_tk_grad = tokens.grad.data.clone()
    o_gt_grad = layer.gate_weight.grad.data.clone()

    # reset all gradients
    tokens.grad.zero_()
    layer.gate_weight.grad.zero_()

    layer.enable_kernel = True
    new_out = layer(tokens)  # get outputs through colossal kernel

    if data_type == torch.float32:
        check_equal(old_out, new_out)
    else:
        check_equal(old_out, new_out, 1e-2)
    # forward function passed

    new_out.backward(grad)  # get new type gradient
    n_tk_grad = tokens.grad.data.clone()
    n_gt_grad = layer.gate_weight.grad.data.clone()

    if data_type == torch.float32:
        check_equal(o_tk_grad, n_tk_grad)
    else:
        check_equal(o_tk_grad, o_tk_grad, 1e-2)
    # tokens gradient is correct

    if data_type == torch.float32:
        check_equal(o_gt_grad, n_gt_grad, 5e-05)
    else:
        check_equal(o_gt_grad, n_gt_grad, 2e-01)
    # bias gradient is correct


@pytest.mark.dist
@pytest.mark.parametrize("rs", [131])
@pytest.mark.parametrize("hidden_size", [32, 144])
@pytest.mark.parametrize("data_type", [torch.float32, torch.float16])
@pytest.mark.parametrize("topk", [1, 2])
@rerun_if_address_is_in_use()
def test_moe_kernel(rs, hidden_size, data_type, topk):
    spawn(run_routing, 4, rs=rs, hidden_size=hidden_size, data_type=data_type, topk=topk)


if __name__ == "__main__":
    test_moe_kernel(2, 256, torch.float16, 2)

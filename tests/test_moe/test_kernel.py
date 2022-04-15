from functools import partial
import pytest
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import colossalai
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import free_port, get_current_device
from colossalai.nn.layer.moe import Top1Router, Top2Router, MoeLayer, Experts
from colossalai.context.moe_context import MOE_CONTEXT
from colossalai.testing import rerun_if_address_is_in_use

BATCH_SIZE = 16
NUM_EXPERTS = 4
CONFIG = dict()


def check_equal(tensor_a, tensor_b, atol=1e-06):
    assert torch.allclose(tensor_a, tensor_b, rtol=0, atol=atol) is True


def run_routing(rank, world_size, port, rs=2, hidden_size=128, data_type=torch.float32, router=Top2Router):
    # Here we do not need TF32, since it brings absolute error on results
    torch.backends.cuda.matmul.allow_tf32 = False

    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    local_rank = gpc.get_local_rank(ParallelMode.GLOBAL)

    MOE_CONTEXT.setup(42)    # MOE environment initialization
    MOE_CONTEXT.reset_loss()
    torch.manual_seed(rs + local_rank)    # set each process has different random seed

    # get randomized data
    tokens = torch.randn(BATCH_SIZE, hidden_size, dtype=data_type, device=get_current_device(), requires_grad=True)

    expert_module = nn.Linear
    expert_factor = dict(in_features=hidden_size, out_features=hidden_size, device=get_current_device())
    expert = Experts(expert_module, NUM_EXPERTS, **expert_factor)
    layer = MoeLayer(hidden_size, NUM_EXPERTS, router(capacity_factor_train=1.0), expert)
    if data_type == torch.float16:
        layer = layer.half()

    # use matrix multiplication instead of COL_MOE_KERNL in MOE dispatch and combine
    layer.use_kernel = False
    old_out = layer(tokens)
    ech = old_out.shape
    grad = torch.randn(ech, device=get_current_device())
    old_out.backward(grad)    # get gradient

    # save all results
    o_tk_grad = tokens.grad.data.clone()
    o_gt_grad = layer.gate.weight.grad.data.clone()

    # reset all gradients
    tokens.grad.zero_()
    layer.gate.weight.grad.zero_()

    layer.use_kernel = True
    new_out = layer(tokens)    # get ouputs through colossal kernel

    if data_type == torch.float32:
        check_equal(old_out, new_out)
    else:
        check_equal(old_out, new_out, 1e-2)
    # forward function passed

    new_out.backward(grad)    # get new type gradient
    n_tk_grad = tokens.grad.data.clone()
    n_gt_grad = layer.gate.weight.grad.data.clone()

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
@pytest.mark.parametrize("router", [Top1Router, Top2Router])
@rerun_if_address_is_in_use()
def test_moe_kernel(rs, hidden_size, data_type, router):
    world_size = 4
    run_func = partial(run_routing,
                       world_size=world_size,
                       port=free_port(),
                       rs=rs,
                       hidden_size=hidden_size,
                       data_type=data_type,
                       router=router)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_moe_kernel(2, 256, torch.float16, Top2Router)

import os

import pytest
import torch

from colossalai.accelerator import get_accelerator
from colossalai.moe._operation import MoeCombine, MoeDispatch, moe_cumsum

NUM_EXPERTS = 4
BATCH_SIZE = 4
SEQ_LEN = 4

MOE_TENSOR_PATH = os.getenv("MOE_TENSOR_PATH")


def check_equal(tensor_a, tensor_b, atol=1e-06):
    assert torch.allclose(tensor_a, tensor_b, rtol=0, atol=atol) is True


def run_moe_cumsum():
    test_mask = torch.tensor(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=torch.int32,
    ).to("cuda")
    out_no_kernel = moe_cumsum(test_mask, use_kernel=False)
    out_kernel = moe_cumsum(test_mask, use_kernel=True)
    print(out_no_kernel.dtype, out_kernel.dtype)
    check_equal(out_no_kernel.to(torch.int32), out_kernel)


def run_moe_dispatch_combine_fwd_bwd(data_type=torch.float32, hidden_size=128, num_experts=4):
    tokens = torch.randn(
        BATCH_SIZE, hidden_size, dtype=data_type, device=get_accelerator().get_current_device(), requires_grad=True
    )

    # use kernel
    route_result_list_kernel = torch.load(f"{MOE_TENSOR_PATH}/True_4_{data_type}.pt")
    # dispatch
    dispatch_data_kernel = MoeDispatch.apply(tokens, *route_result_list_kernel[1:])
    dispatch_data_kernel = dispatch_data_kernel.reshape(num_experts, -1, hidden_size)
    # combine
    expert_output = dispatch_data_kernel.reshape(-1, hidden_size)
    ans_kernel = MoeCombine.apply(expert_output, *route_result_list_kernel)

    # no kernel
    route_result_list_no_kernel = torch.load(f"{MOE_TENSOR_PATH}/False_2_{data_type}.pt")
    # dispatch
    sec_mask_f = route_result_list_no_kernel[1].type_as(tokens)
    dispatch_data_no_kernel = torch.matmul(sec_mask_f.permute(1, 2, 0), tokens)
    # combine
    combine_weights = route_result_list_no_kernel[0].type_as(tokens)
    combine_weights = combine_weights.view(combine_weights.shape[0], -1)
    expert_output = expert_output.view(-1, expert_output.shape[-1])
    ans_no_kernel = torch.matmul(combine_weights, expert_output)

    # check fwd
    if data_type == torch.float32:
        check_equal(dispatch_data_kernel.reshape(dispatch_data_no_kernel.shape), dispatch_data_no_kernel)
    else:
        check_equal(dispatch_data_kernel.reshape(dispatch_data_no_kernel.shape), dispatch_data_no_kernel, 1e-2)

    if data_type == torch.float32:
        check_equal(ans_kernel, ans_no_kernel)
    else:
        check_equal(ans_kernel, ans_no_kernel, 1e-2)

    # check bwd
    out_shape = ans_kernel.shape
    grad = torch.randn(out_shape, device=get_accelerator().get_current_device())

    ans_kernel.backward(grad, retain_graph=True)
    grad_kernel = tokens.grad.data.clone()
    tokens.grad.zero_()

    ans_no_kernel.backward(grad)  # get gradient
    grad_no_kernel = tokens.grad.data.clone()
    tokens.grad.zero_()

    if data_type == torch.float32:
        check_equal(grad_no_kernel, grad_kernel)
    else:
        check_equal(grad_no_kernel, grad_kernel, 1e-2)


@pytest.mark.parametrize("data_type", [torch.float32, torch.float16])
def test_moe_kernel(data_type):
    torch.manual_seed(1024)
    run_moe_cumsum()
    run_moe_dispatch_combine_fwd_bwd(data_type=data_type)

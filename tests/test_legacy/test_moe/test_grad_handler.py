import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.legacy.moe.manager import MOE_MANAGER

# from colossalai.shardformer.layer.moe.layers import SparseMLP
from colossalai.testing import assert_equal_in_group, rerun_if_address_is_in_use, spawn
from tests.test_moe.moe_utils import MoeGradientHandler

BATCH_SIZE = 4
DIM = 16


def run_test(rank, world_size, port):
    colossalai.launch(
        rank=rank,
        world_size=world_size,
        host="localhost",
        port=port,
        backend="nccl",
    )

    MOE_MANAGER.setup(parallel="EP")  # MOE initialization
    num_experts_list = [1, 2, 4]
    layer_list = []
    for num_experts in num_experts_list:
        moe_layer = SparseMLP(
            hidden_size=DIM,
            intermediate_size=DIM * 4,
            num_experts=num_experts,
            router_top_k=1,
            router_noisy_policy="Jitter",
        )
        layer_list.append(moe_layer)

    model = nn.ModuleList(layer_list)
    model = model.to(get_accelerator().get_current_device())
    dist_dict = MOE_MANAGER.parallel_info_dict
    assert_equal_in_group(layer_list[0].experts.wi.data, dist_dict[1].dp_group)
    assert_equal_in_group(layer_list[0].experts.wo.data, dist_dict[1].dp_group)
    assert_equal_in_group(layer_list[1].experts.wi.data, dist_dict[2].dp_group)
    assert_equal_in_group(layer_list[1].experts.wo.data, dist_dict[2].dp_group)
    assert_equal_in_group(layer_list[2].experts.wi.data, dist_dict[4].dp_group)
    assert_equal_in_group(layer_list[2].experts.wo.data, dist_dict[4].dp_group)
    # MoE model synchronization passed

    grad_handler = MoeGradientHandler(model, 0)

    rank = dist.get_rank()
    torch.cuda.manual_seed(78 + rank)
    data = torch.randn(BATCH_SIZE, DIM, device=get_accelerator().get_current_device())
    grad = torch.randn_like(data)

    MOE_MANAGER.reset_loss()
    for layer in layer_list:
        data = layer(data)
    data.backward(grad)
    grad_handler.handle_gradient()

    assert_equal_in_group(layer_list[0].experts.wi.grad, dist_dict[1].dp_group)
    assert_equal_in_group(layer_list[0].experts.wo.grad, dist_dict[1].dp_group)
    assert_equal_in_group(layer_list[1].experts.wi.grad, dist_dict[2].dp_group)
    assert_equal_in_group(layer_list[1].experts.wo.grad, dist_dict[2].dp_group)
    assert_equal_in_group(layer_list[2].experts.wi.grad, dist_dict[4].dp_group)
    assert_equal_in_group(layer_list[2].experts.wo.grad, dist_dict[4].dp_group)
    # MoE grad handler test passed


@pytest.mark.skip(reason="moe need to be refactored")
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_grad_handler():
    spawn(run_test, 4)


if __name__ == "__main__":
    test_grad_handler()

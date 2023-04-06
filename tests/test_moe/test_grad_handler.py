import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

import colossalai
from colossalai.context.moe_context import MOE_CONTEXT
from colossalai.engine.gradient_handler import MoeGradientHandler
from colossalai.nn.layer.moe import Experts, MoeLayer, Top1Router, UniformNoiseGenerator
from colossalai.testing import assert_equal_in_group, rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device
from colossalai.utils.moe import sync_moe_model_param

BATCH_SIZE = 4
DIM = 16
CONFIG = dict()


def run_test(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    expert_module = nn.Linear
    expert_factor = dict(in_features=DIM, out_features=DIM, device=get_current_device())

    MOE_CONTEXT.setup(42)    # MOE initialization
    noisy_func = UniformNoiseGenerator()
    router = Top1Router(noisy_func=noisy_func)
    num_experts_list = [1, 2, 4]
    layer_list = []
    for num_experts in num_experts_list:
        exp = Experts(expert_module, num_experts, **expert_factor)
        moe_layer = MoeLayer(DIM, num_experts, router, exp)
        layer_list.append(moe_layer)

    model = nn.ModuleList(layer_list)
    model = model.to(get_current_device())
    sync_moe_model_param(model)

    dist_dict = MOE_CONTEXT.parallel_info_dict
    assert_equal_in_group(layer_list[0].experts.experts[0].weight.data, dist_dict[1].dp_group)
    assert_equal_in_group(layer_list[1].experts.experts[0].weight.data, dist_dict[2].dp_group)
    # MoE model synchronization passed

    grad_handler = MoeGradientHandler(model, 0)

    rank = dist.get_rank()
    torch.cuda.manual_seed(78 + rank)
    data = torch.randn(BATCH_SIZE, DIM, device=get_current_device())
    grad = torch.randn_like(data)

    MOE_CONTEXT.reset_loss()
    for layer in layer_list:
        data, _ = layer(data)
    data.backward(grad)
    grad_handler.handle_gradient()

    assert_equal_in_group(layer_list[0].experts.experts[0].weight.grad, dist_dict[1].dp_group)
    assert_equal_in_group(layer_list[0].experts.experts[0].bias.grad, dist_dict[1].dp_group)

    assert_equal_in_group(layer_list[1].experts.experts[0].weight.grad, dist_dict[2].dp_group)
    assert_equal_in_group(layer_list[1].experts.experts[0].bias.grad, dist_dict[2].dp_group)
    # MoE grad handler test passed


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_grad_handler():
    spawn(run_test, 4)


if __name__ == '__main__':
    test_grad_handler()

from functools import partial
import pytest
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import colossalai
from colossalai.utils import free_port, get_current_device
from colossalai.nn.layer.moe import Top1Router, UniformNoiseGenerator, MoeLayer, Experts
from colossalai.core import moe_context as moe_env
from colossalai.utils import sync_moe_model_param
from colossalai.engine.gradient_handler import MoeGradientHandler
from colossalai.testing import assert_equal_in_group

BATCH_SIZE = 4
DIM = 16
CONFIG = dict()


def my_test(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    expert_module = nn.Linear
    expert_factor = dict(in_features=DIM, out_features=DIM, device=get_current_device())

    moe_env.setup(42)
    noisy_func = UniformNoiseGenerator()
    router = Top1Router(noisy_func=noisy_func)
    exp0 = Experts(expert_module, 1, **expert_factor)
    layer1 = MoeLayer(DIM, 1, router, exp0)
    exp1 = Experts(expert_module, 2, **expert_factor)
    layer2 = MoeLayer(DIM, 2, router, exp1)
    exp2 = Experts(expert_module, 4, **expert_factor)
    layer3 = MoeLayer(DIM, 4, router, exp2)

    model = nn.Sequential(layer1, layer2, layer3)
    model = model.to(get_current_device())
    sync_moe_model_param(model)

    dist_dict = moe_env.information
    assert_equal_in_group(exp0.experts[0].weight.data, dist_dict[1].dp_group)
    assert_equal_in_group(exp1.experts[0].weight.data, dist_dict[2].dp_group)
    # MoE model synchronization passed

    grad_handler = MoeGradientHandler(model, 0)

    rank = dist.get_rank()
    torch.cuda.manual_seed(78 + rank)
    data = torch.randn(BATCH_SIZE, DIM, device=get_current_device())
    grad = torch.randn_like(data)

    moe_env.reset_loss()
    outputs = model(data)
    outputs.backward(grad)
    grad_handler.handle_gradient()

    assert_equal_in_group(exp0.experts[0].weight.grad, dist_dict[1].dp_group)
    assert_equal_in_group(exp1.experts[0].weight.grad, dist_dict[2].dp_group)
    # MoE grad handler test passed


@pytest.mark.dist
def test_moe_group():
    world_size = 4
    run_func = partial(my_test, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_moe_group()

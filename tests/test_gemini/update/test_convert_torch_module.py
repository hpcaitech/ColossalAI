import os
from functools import partial

import pytest
import torch
import torch.multiprocessing as mp

import colossalai
from colossalai.nn.parallel.utils import convert_to_torch_module
from colossalai.tensor import ColoTensor
from colossalai.testing import parameterize, rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.utils.cuda import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
from tests.components_to_test.registry import non_distributed_component_funcs


@parameterize('model_name', ['resnet18', 'bert'])
def run_convert_torch_module(model_name: str):
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, _, _, _, _ = get_components_func()

    with ColoInitContext(device='cpu'):
        model = model_builder(checkpoint=False)

    from colossalai.nn.parallel import GeminiDDP
    model = GeminiDDP(model, device=get_current_device(), placement_policy='auto', pin_memory=True)

    pytorch_model = convert_to_torch_module(model)

    for n, p in pytorch_model.named_parameters():
        assert not isinstance(p, ColoTensor)


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_convert_torch_module()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_convert_torch_module(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_convert_torch_module(2)

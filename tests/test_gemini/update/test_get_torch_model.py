import os
from functools import partial

import pytest
import torch
import torch.multiprocessing as mp

import colossalai
from colossalai.nn.parallel import GeminiDDP
from colossalai.nn.parallel.utils import get_static_torch_model
from colossalai.tensor import ColoParameter
from colossalai.testing import parameterize, rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.utils.cuda import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
from tests.components_to_test.registry import non_distributed_component_funcs


@parameterize('model_name', ['hanging_param_model', 'resnet18', 'gpt2'])
def run_convert_torch_module(model_name: str):
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, _, _, _, _ = get_components_func()

    with ColoInitContext(device=torch.device("cpu")):
        model = model_builder(checkpoint=False)
    model = GeminiDDP(model, device=get_current_device(), placement_policy='auto', pin_memory=True)
    pytorch_model = get_static_torch_model(model, only_rank_0=False)

    for n, p in pytorch_model.named_parameters():
        assert type(p) == torch.nn.Parameter, f"type error: {n} is a {type(p)}"

    # get the static model should not change the original model
    for n, p in model.named_parameters():
        assert isinstance(p, ColoParameter)

    for (pn, pm), (cn, cm) in zip(pytorch_model.named_modules(), model.named_modules()):
        assert pn == cn
        assert id(pm) != id(cm)
        for pp, cp in zip(pm.parameters(recurse=False), cm.parameters(recurse=False)):
            assert id(pp) != id(cp)
            assert pp.shape == cp.shape


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

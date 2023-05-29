from copy import deepcopy

import torch

from colossalai.elixir.cuda import gpu_device
from colossalai.elixir.search import simple_search
from colossalai.testing import run_on_environment_flag
from tests.test_elixir.utils import TEST_MODELS


def step_fn(model, inp):
    model(**inp).backward()


@run_on_environment_flag('ELX')
def test_simple_search():
    model_fn, data_fn = TEST_MODELS.get('small')
    model = model_fn()
    data = data_fn()

    sr = simple_search(model,
                       1,
                       split_number=5,
                       shard_device=gpu_device(),
                       prefetch=True,
                       verbose=True,
                       inp=data,
                       step_fn=step_fn)

    chunk_plans = deepcopy(sr.param_chunk_plans)
    private_plan = chunk_plans.pop(0)
    assert private_plan.name_list == ['embed.weight']
    assert private_plan.chunk_size == 320
    assert private_plan.kwargs.get('shard_device') == gpu_device()

    assert chunk_plans[0].name_list == ['norm1.weight', 'norm1.bias']
    assert chunk_plans[1].name_list == ['mlp.proj1.weight', 'mlp.proj1.bias']
    assert chunk_plans[2].name_list == ['mlp.proj2.weight', 'mlp.proj2.bias']
    assert chunk_plans[3].name_list == ['norm2.weight']
    assert chunk_plans[4].name_list == ['norm2.bias']

    for plan in chunk_plans:
        assert plan.chunk_size == 1088
        assert plan.kwargs.get('shard_device') == gpu_device()


if __name__ == '__main__':
    test_simple_search()

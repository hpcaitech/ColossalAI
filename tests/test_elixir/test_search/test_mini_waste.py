from copy import deepcopy

import torch

from colossalai.elixir.cuda import gpu_device
from colossalai.elixir.search import minimum_waste_search
from colossalai.testing import run_on_environment_flag
from tests.test_elixir.utils import TEST_MODELS


def step_fn(model, inp):
    model(**inp).backward()


@run_on_environment_flag('ELX')
def test_mini_waste_search():
    model_fn, data_fn = TEST_MODELS.get('gpt2_small')
    model = model_fn()
    data = data_fn()

    sr = minimum_waste_search(model,
                              1,
                              unified_dtype=torch.float16,
                              cpu_offload=True,
                              prefetch=True,
                              verbose=True,
                              inp=data,
                              step_fn=step_fn)

    chunk_plans = deepcopy(sr.param_chunk_plans)
    for plan in chunk_plans:
        assert plan.chunk_dtype == torch.float16
        assert plan.kwargs.get('shard_device') == torch.device('cpu')
        assert plan.kwargs.get('cpu_pin_memory') == True


if __name__ == '__main__':
    test_mini_waste_search()

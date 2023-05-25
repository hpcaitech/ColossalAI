import pytest
import torch

from colossalai.elixir.tracer.memory_tracer import cuda_memory_profiling
from colossalai.testing import run_on_environment_flag
from tests.test_elixir.utils import TEST_MODELS, to_cuda


def one_step(model, inp):
    loss = model(**inp)
    loss.backward()
    return loss


def try_one_model(model_fn, data_fn):
    model = model_fn().cuda()
    data = to_cuda(data_fn())
    one_step(model, data)    # generate gradients

    pre_cuda_alc = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    one_step(model, data)
    aft_cuda_alc = torch.cuda.max_memory_allocated()
    torch_activation_occ = aft_cuda_alc - pre_cuda_alc
    model.zero_grad(set_to_none=True)
    print('normal', torch_activation_occ)

    before = torch.cuda.memory_allocated()
    profiling_dict = cuda_memory_profiling(model, data, one_step)
    after = torch.cuda.memory_allocated()
    print('profiling', profiling_dict)
    assert before == after
    assert torch_activation_occ == profiling_dict['activation_occ']
    print('Check is ok.')


@run_on_environment_flag('ELX')
def test_cuda_profiler():
    model_list = ['resnet', 'gpt2_micro']
    for name in model_list:
        model_fn, data_fn = TEST_MODELS.get(name)
        try_one_model(model_fn, data_fn)


if __name__ == '__main__':
    test_cuda_profiler()

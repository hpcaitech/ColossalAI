import torch
import colossalai
import copy
import pytest
import torch.multiprocessing as mp
from colossalai.amp import convert_to_naive_amp
from tests.components_to_test.registry import non_distributed_component_funcs
from colossalai.testing import assert_close_loose
from colossalai.utils import free_port
from functools import partial


def check_equal(a, b):
    """
    This function checks if two tensors are equal within tolerance
    """
    assert torch.allclose(a.float(), b.float(), rtol=1e-4, atol=1e-3), f'a = {a}, b = {b}'


def run_naive_amp():
    """
    In this test, we compare the naive fp16 optimizer implemented in colossalai 
    and fp32 torch optimizer
    """

    # create layer
    test_models = ['repeated_computed_layers', 'nested_model']
    for test_name in test_models:
        get_component_func = non_distributed_component_funcs.get_callable(test_name)
        model_builder, train_dataloader, _, optim_class, _ = get_component_func()

        # create model
        amp_model = model_builder(checkpoint=True).cuda()
        torch_model = copy.deepcopy(amp_model)

        # create optimizer
        amp_optimizer = optim_class(amp_model.parameters(), lr=1e-3)
        torch_optimizer = optim_class(torch_model.parameters(), lr=1e-3)

        # inject naive amp
        amp_config = dict(initial_scale=1)
        amp_model, amp_optimizer = convert_to_naive_amp(amp_model, amp_optimizer, amp_config)

        # create data
        data_iter = iter(train_dataloader)
        data, label = next(data_iter)
        data = data.cuda()

        # forward pass
        amp_output = amp_model(data)
        torch_output = torch_model(data)
        assert_close_loose(amp_output, torch_output)

        # backward
        amp_optimizer.backward(amp_output.mean())
        torch_output.mean().backward()

        # check grad
        for amp_param, torch_param in zip(amp_model.parameters(), torch_model.parameters()):
            assert_close_loose(amp_param.grad, torch_param.grad.half())

        # step
        amp_optimizer.step()
        torch_optimizer.step()

        # check updated param
        for amp_param, torch_param in zip(amp_model.parameters(), torch_model.parameters()):
            assert_close_loose(amp_param, torch_param.half())


def run_dist(rank, world_size, port):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')
    run_naive_amp()


@pytest.mark.dist
def test_naive_amp():
    world_size = 1
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_naive_amp()

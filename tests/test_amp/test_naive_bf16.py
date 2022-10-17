import torch
import colossalai
import torch.multiprocessing as mp
from colossalai.amp import convert_to_naive_amp
from tests.components_to_test.registry import non_distributed_component_funcs
from colossalai.testing import assert_close_loose, rerun_if_address_is_in_use
from colossalai.utils import free_port


import copy
import pytest
from functools import partial


def check_equal(a, b):
    """
    This function checks if two tensors are equal within tolerance
    """
    assert torch.allclose(a.float(), b.float(), rtol=1e-4, atol=1e-3), f'a = {a}, b = {b}'


def run_naive_bf16_amp():
    """
    In this test, we compare the naive fp16 optimizer implemented in colossalai 
    and fp32 torch optimizer
    """

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # create layer
    test_models = ['repeated_computed_layers', 'nested_model', 'resnet18']
    for test_name in test_models:
        get_component_func = non_distributed_component_funcs.get_callable(test_name)
        model_builder, train_dataloader, _, optim_class, _ = get_component_func()

        # create model
        naive_fp16_amp_model = model_builder(checkpoint=True).cuda()
        naive_bf16_amp_model = copy.deepcopy(naive_fp16_amp_model)

        # create optimizer
        naive_bf16_amp_optimizer = optim_class(naive_bf16_amp_model.parameters(), lr=1e-3)
        naive_fp16_amp_optimizer = optim_class(naive_fp16_amp_model.parameters(), lr=1e-3)

        # inject naive and naive_fp16 amp
        naive_bf16_amp_config = dict(initial_scale=1)
        naive_bf16_amp_model, naive_bf16_amp_optimizer = convert_to_naive_amp(naive_bf16_amp_model, naive_bf16_amp_optimizer, precision_type=torch.bfloat16,
                                                                              amp_config=naive_bf16_amp_config)
        naive_fp16_amp_config = dict(initial_scale=1)
        naive_fp16_amp_model, naive_fp16_amp_optimizer = convert_to_naive_amp(naive_fp16_amp_model, naive_fp16_amp_optimizer, naive_fp16_amp_config)

        # create data
        data_iter = iter(train_dataloader)
        data, label = next(data_iter)
        data = data.cuda()

        # forward pass
        naive_bf16_amp_output = naive_bf16_amp_model(data)
        naive_fp16_amp_output = naive_fp16_amp_model(data)
        assert_close_loose(naive_bf16_amp_output, naive_fp16_amp_output, 1e-1, 1e-1)

        # backward
        naive_bf16_amp_optimizer.backward(naive_bf16_amp_output.mean())
        naive_fp16_amp_optimizer.backward(naive_fp16_amp_output.mean())

        # check grad
        for naive_bf16_amp_param, naive_fp16_amp_param in zip(naive_bf16_amp_model.parameters(), naive_fp16_amp_model.parameters()):
            assert_close_loose(naive_bf16_amp_param.grad, naive_fp16_amp_param.grad, 1, 1)

        # step
        naive_bf16_amp_optimizer.step()
        naive_fp16_amp_optimizer.step()

        # check updated param
        for naive_bf16_amp_param, naive_fp16_amp_param in zip(naive_bf16_amp_model.parameters(), naive_fp16_amp_model.parameters()):
            assert_close_loose(naive_bf16_amp_param, naive_fp16_amp_param, 1e-2, 1e-2)


def run_dist(rank, world_size, port):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')
    run_naive_bf16_amp()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_naive_bf16_amp():
    world_size = 1
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_naive_bf16_amp()
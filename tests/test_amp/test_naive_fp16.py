import copy

import pytest
import torch

import colossalai
from colossalai.amp import convert_to_apex_amp, convert_to_naive_amp
from colossalai.testing import assert_close_loose, clear_cache_before_run, rerun_if_address_is_in_use, spawn
from tests.components_to_test.registry import non_distributed_component_funcs


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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # create layer
    test_models = ['repeated_computed_layers', 'nested_model', 'resnet18']
    for test_name in test_models:
        get_component_func = non_distributed_component_funcs.get_callable(test_name)
        model_builder, train_dataloader, _, optim_class, _ = get_component_func()

        # create model
        naive_amp_model = model_builder(checkpoint=True).cuda()
        apex_amp_model = copy.deepcopy(naive_amp_model)

        # create optimizer
        # we use SGD here, since the correctness of gradient clipping can't be tested with Adam
        naive_amp_optimizer = torch.optim.SGD(naive_amp_model.parameters(), lr=1e-3)
        apex_amp_optimizer = torch.optim.SGD(apex_amp_model.parameters(), lr=1e-3)

        # inject naive and apex amp
        naive_amp_config = dict(initial_scale=128, clip_grad_norm=1.0)
        naive_amp_model, naive_amp_optimizer = convert_to_naive_amp(naive_amp_model, naive_amp_optimizer,
                                                                    naive_amp_config)
        apex_amp_config = dict(opt_level='O2', loss_scale=128, keep_batchnorm_fp32=False)
        apex_amp_model, apex_amp_optimizer = convert_to_apex_amp(apex_amp_model, apex_amp_optimizer, apex_amp_config)

        # create data
        data_iter = iter(train_dataloader)
        data, label = next(data_iter)
        data = data.cuda()

        # forward pass
        naive_amp_output = naive_amp_model(data)
        apex_amp_output = apex_amp_model(data)
        assert_close_loose(naive_amp_output, apex_amp_output)

        # backward
        # use sum() to get big gradient
        naive_amp_optimizer.backward(naive_amp_output.sum())
        apex_amp_optimizer.backward(apex_amp_output.sum())

        # check grad
        for naive_amp_param, apex_amp_param in zip(naive_amp_model.parameters(), apex_amp_model.parameters()):
            assert_close_loose(naive_amp_param.grad, apex_amp_param.grad)

        # clip gradient
        apex_amp_optimizer.clip_grad_norm(model=apex_amp_model, max_norm=1.0)

        # step
        naive_amp_optimizer.step()
        apex_amp_optimizer.step()

        # check updated param
        for naive_amp_param, apex_amp_param in zip(naive_amp_model.parameters(), apex_amp_model.parameters()):
            assert_close_loose(naive_amp_param, apex_amp_param)


def run_dist(rank, world_size, port):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')
    run_naive_amp()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_naive_amp():
    spawn(run_dist, 1)


if __name__ == '__main__':
    test_naive_amp()

import copy

import pytest
import torch

import colossalai
from colossalai.amp import convert_to_apex_amp, convert_to_torch_amp
from colossalai.testing import assert_close_loose, clear_cache_before_run, rerun_if_address_is_in_use, spawn
from tests.components_to_test.registry import non_distributed_component_funcs


def run_torch_amp():
    """
    In this test, we compare the torch amp and apex amp implemented in colossalai
    """

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # create layer
    test_models = ['resnet18', 'simple_net']
    for test_name in test_models:
        get_component_func = non_distributed_component_funcs.get_callable(test_name)
        model_builder, train_dataloader, _, optim_class, _ = get_component_func()

        # create model
        torch_amp_model = model_builder(checkpoint=True).cuda()
        apex_amp_model = copy.deepcopy(torch_amp_model)

        # create optimizer
        # we use SGD here, since the correctness of gradient clipping can't be tested with Adam
        torch_amp_optimizer = torch.optim.SGD(torch_amp_model.parameters(), lr=1e-3)
        apex_amp_optimizer = torch.optim.SGD(apex_amp_model.parameters(), lr=1e-3)

        # inject torch and apex amp
        torch_amp_config = dict(init_scale=128, enabled=True)
        torch_amp_model, torch_amp_optimizer, _ = convert_to_torch_amp(torch_amp_model,
                                                                       torch_amp_optimizer,
                                                                       amp_config=torch_amp_config)
        apex_amp_config = dict(opt_level='O1', loss_scale=128)
        apex_amp_model, apex_amp_optimizer = convert_to_apex_amp(apex_amp_model, apex_amp_optimizer, apex_amp_config)

        # create data
        data_iter = iter(train_dataloader)
        data, label = next(data_iter)
        data = data.cuda()

        # forward pass
        torch_amp_output = torch_amp_model(data)
        apex_amp_output = apex_amp_model(data)
        assert_close_loose(torch_amp_output, apex_amp_output)

        for torch_amp_param, apex_amp_param in zip(torch_amp_model.parameters(), apex_amp_model.parameters()):
            assert_close_loose(torch_amp_param, apex_amp_param)

        # backward
        # use sum() to get big gradient
        torch_amp_optimizer.backward(torch_amp_output.sum())
        apex_amp_optimizer.backward(apex_amp_output.sum())

        # check grad
        # In apex amp, grad is not scaled before backward, but torch amp does
        for torch_amp_param, apex_amp_param in zip(torch_amp_model.parameters(), apex_amp_model.parameters()):
            assert_close_loose(torch_amp_param.grad, apex_amp_param.grad * apex_amp_config['loss_scale'])

        # clip gradient
        apex_amp_optimizer.clip_grad_norm(model=apex_amp_model, max_norm=1.0)
        torch_amp_optimizer.clip_grad_norm(model=torch_amp_model, max_norm=1.0)

        # step
        torch_amp_optimizer.step()
        apex_amp_optimizer.step()

        # check updated param and grad
        for torch_amp_param, apex_amp_param in zip(torch_amp_model.parameters(), apex_amp_model.parameters()):
            assert_close_loose(torch_amp_param.grad, apex_amp_param.grad)
            assert_close_loose(torch_amp_param, apex_amp_param)


def run_dist(rank, world_size, port):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')
    run_torch_amp()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_torch_amp():
    spawn(run_dist, 1)


if __name__ == '__main__':
    test_torch_amp()

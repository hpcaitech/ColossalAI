from functools import partial

import pytest
import torch
import torch.multiprocessing as mp

import colossalai
from colossalai.gemini.memory_tracer import MemtracerWrapper
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from tests.components_to_test import run_fwd_bwd
from tests.components_to_test.registry import non_distributed_component_funcs


def run_tracer(rank, world_size, port, use_grad_check=True):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    test_models = ['repeated_computed_layers', 'resnet18', 'hanging_param_model', 'bert']
    # test_models = ['bert']
    for model_name in test_models:
        get_components_func = non_distributed_component_funcs.get_callable(model_name)
        model_builder, train_dataloader, _, _, criterion = get_components_func()

        # init model on cpu
        # TODO() memtrace hook can not handle buff registered on a non-leaf module (for example the BertEmbedding).
        # a simple method is that always puts buff on cuda and viewed them as non-model data.
        model = MemtracerWrapper(model_builder(checkpoint=use_grad_check))

        for n, buff in model.named_buffers():
            buff.data = buff.data.cuda()
        for i, (data, label) in enumerate(train_dataloader):
            if i > 1:
                break
            data = data.cuda()
            label = label.cuda()

            run_fwd_bwd(model, data, label, criterion)

        model._ophook_list[0].print_non_model_data()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1])
@pytest.mark.parametrize("use_grad_check", [True, False])
@rerun_if_address_is_in_use()
def test_tracer(world_size, use_grad_check):
    run_func = partial(run_tracer, world_size=world_size, port=free_port(), use_grad_check=use_grad_check)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_tracer(1, True)

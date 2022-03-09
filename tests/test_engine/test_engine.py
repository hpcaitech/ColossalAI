from functools import partial

import colossalai
import pytest
import torch.multiprocessing as mp
from colossalai.amp import AMP_TYPE
from colossalai.context import Config
from colossalai.core import global_context as gpc
from colossalai.utils import free_port
from tests.components_to_test.registry import non_distributed_component_funcs

CONFIG = dict(parallel=dict(pipeline=dict(size=1), tensor=dict(size=1, mode=None)),
              fp16=dict(mode=None),
              clip_grad_norm=1.0)


def run_train():
    test_models = ['repeated_computed_layers', 'resnet18', 'repeated_computed_layers']
    # FIXME: test bert
    for model_name in test_models:
        get_components_func = non_distributed_component_funcs.get_callable(model_name)
        model_builder, train_dataloader, _, optimizer_builder, criterion = get_components_func()

        model = model_builder(checkpoint=False)
        engine, train_dataloader, *args = colossalai.initialize(model=model,
                                                                optimizer=optimizer_builder(model),
                                                                criterion=criterion,
                                                                train_dataloader=train_dataloader)

        try:
            engine.train()
            for data, label in train_dataloader:
                engine.zero_grad()
                data = data.cuda()
                label = label.cuda()
                if criterion:
                    output = engine(data)
                    loss = engine.criterion(output, label)
                else:
                    loss = engine(data, label)
                engine.backward(loss)
                engine.step()
                break
        except IndexError:
            # if using apex amp, NetWithRepeatedlyComputedLayers will raise an index out of range issue
            # the following check fails in apex
            # if cached_x.grad_fn.next_functions[1][0].variable is not x:
            continue


def run_with_no_amp():
    run_train()


def run_with_torch_amp():
    # hack config
    CONFIG['fp16']['mode'] = AMP_TYPE.TORCH
    gpc._config = Config(CONFIG)
    run_train()


def run_with_apex_amp():
    # hack config
    CONFIG['fp16']['mode'] = AMP_TYPE.APEX
    gpc._config = Config(CONFIG)
    run_train()


def run_with_naive_amp():
    # hack config
    CONFIG['fp16']['mode'] = AMP_TYPE.NAIVE
    gpc._config = Config(CONFIG)
    run_train()


def run_engine(rank, world_size, port):
    # init dist env
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_with_no_amp()
    run_with_torch_amp()
    run_with_apex_amp()
    run_with_naive_amp()


@pytest.mark.dist
def test_engine():
    world_size = 4
    run_func = partial(run_engine, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_engine()

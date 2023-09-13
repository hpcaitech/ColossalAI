import pytest

import colossalai
from colossalai.amp import AMP_TYPE
from colossalai.core import global_context as gpc
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from tests.components_to_test.registry import non_distributed_component_funcs

CONFIG = dict(parallel=dict(pipeline=dict(size=1), tensor=dict(size=1, mode=None)),
              fp16=dict(mode=None),
              clip_grad_norm=1.0)


@parameterize('model_name', ['repeated_computed_layers', 'resnet18', 'repeated_computed_layers'])
@parameterize('amp_mode', [AMP_TYPE.APEX, AMP_TYPE.TORCH, AMP_TYPE.NAIVE, None])
def run_train(model_name, amp_mode):
    # FIXME: test bert
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    gpc.config.fp16['mode'] = amp_mode
    model_builder, train_dataloader, _, optimizer_class, criterion = get_components_func()

    model = model_builder(checkpoint=False)
    engine, train_dataloader, *args = colossalai.initialize(model=model,
                                                            optimizer=optimizer_class(model.parameters(), lr=1e-3),
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
        pass


def run_engine(rank, world_size, port):
    # init dist env
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_train()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_engine():
    spawn(run_engine, 2)


if __name__ == '__main__':
    test_engine()

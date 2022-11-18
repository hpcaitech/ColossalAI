import torch
import torch.nn as nn

import colossalai
from colossalai.gemini.memory_tracer import MemtracerWrapper
from tests.components_to_test.registry import non_distributed_component_funcs


def run_fwd_bwd(model, data, label, criterion, enable_autocast=False):
    with torch.cuda.amp.autocast(enabled=enable_autocast):
        if criterion:
            y = model(data)
            loss = criterion(y, label)
        else:
            loss = model(data, label)
        loss = loss.float()
    model.backward(loss)


def test_tracer():
    # reset the manager, in case that there exists memory information left
    test_models = ['repeated_computed_layers', 'resnet18', 'no_leaf_module']
    for model_name in test_models:
        get_components_func = non_distributed_component_funcs.get_callable(model_name)
        model_builder, train_dataloader, _, _, criterion = get_components_func()

        # init model on cpu
        model = MemtracerWrapper(model_builder())

        for i, (data, label) in enumerate(train_dataloader):
            if i > 1:
                break
            data = data.cuda()
            label = label.cuda()

            run_fwd_bwd(model, data, label, criterion, False)

        # model._ophook_list[0].print_non_model_data()


if __name__ == '__main__':
    test_tracer()

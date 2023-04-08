import copy

import torch

from colossalai.testing import clear_cache_before_run
from colossalai.zero.legacy.gemini.paramhooks import BaseParamHookMgr
from tests.components_to_test.registry import non_distributed_component_funcs


def allclose(tensor_a: torch.Tensor, tensor_b: torch.Tensor, loose=False) -> bool:
    if loose:
        return torch.allclose(tensor_a, tensor_b, atol=1e-3, rtol=1e-3)
    return torch.allclose(tensor_a, tensor_b)


def run_model(model, inputs, label, criterion, use_param_hook=False):
    if use_param_hook:

        class HooKWrapper:

            def __init__(self) -> None:
                self.hook_triggered_times = 0

            def wrapper_func(self):

                def hook(param, grad) -> torch.Tensor or None:
                    self.hook_triggered_times += 1
                    return grad

                return hook

        hookwrapper = HooKWrapper()
        param_list = [p for p in model.parameters()]
        hook_mgr = BaseParamHookMgr(param_list)
        hook_mgr.register_backward_hooks(hookwrapper.wrapper_func())

    model.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast():
        if criterion:
            y = model(inputs)
            loss = criterion(y, label)
        else:
            loss = model(inputs, label)
        loss = loss.float()
    loss.backward()

    if use_param_hook:
        hook_mgr.remove_hooks()
        return hookwrapper.hook_triggered_times


@clear_cache_before_run()
def test_base_param_hook():
    test_models = ['repeated_computed_layers', 'resnet18', 'hanging_param_model', 'inline_op_model']
    # test_models = ['bert']

    for model_name in test_models:
        get_components_func = non_distributed_component_funcs.get_callable(model_name)
        model_builder, train_dataloader, _, _, criterion = get_components_func()

        torch.manual_seed(0)
        model = model_builder(checkpoint=True).cuda()
        model.train()

        for i, (inputs, label) in enumerate(train_dataloader):
            if i > 0:
                break
            model_copy = copy.deepcopy(model)

            run_model(model, inputs.cuda(), label.cuda(), criterion, False)
            ret2 = run_model(model_copy, inputs.cuda(), label.cuda(), criterion, True)

        # Make sure param hook has only be fired once in case of parameter sharing
        assert ret2 == len(list(model.parameters()))

        for p, p_copy in zip(model.parameters(), model_copy.parameters()):
            assert allclose(p.grad, p_copy.grad), f"{p.grad} vs {p_copy.grad}"


if __name__ == '__main__':
    test_base_param_hook()

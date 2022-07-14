from colossalai.gemini.paramhooks import BaseParamHookMgr
from torch import nn
import torch
import torch.nn.functional as F
import copy


class SubNet(nn.Module):

    def __init__(self, out_features) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, weight):
        return F.linear(x, weight, self.bias)


class Net(nn.Module):

    def __init__(self, checkpoint=False) -> None:
        super().__init__()
        self.fc1 = nn.Linear(5, 5)
        self.sub_fc = SubNet(5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sub_fc(x, self.fc1.weight)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def net_data():
    return (torch.randn(2, 5, dtype=torch.float, device='cuda'),)


def allclose(tensor_a: torch.Tensor, tensor_b: torch.Tensor, loose=False) -> bool:
    if loose:
        return torch.allclose(tensor_a, tensor_b, atol=1e-3, rtol=1e-3)
    return torch.allclose(tensor_a, tensor_b)


def test_base_param_hook():
    torch.manual_seed(0)
    model = Net(checkpoint=True).cuda()
    model.train()
    inputs = net_data()

    def run_model(model, inputs, use_param_hook=False):
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
            y = model(*inputs)
            loss = y.sum()
        loss.backward()

        if use_param_hook:
            hook_mgr.remove_hooks()
            return hookwrapper.hook_triggered_times

    model_copy = copy.deepcopy(model)

    run_model(model, inputs, False)
    ret2 = run_model(model_copy, inputs, True)

    # Make sure param hook has only be fired once in case of parameter sharing
    assert ret2 == len(list(model.parameters()))

    for p, p_copy in zip(model.parameters(), model_copy.parameters()):
        assert allclose(p.grad, p_copy.grad), f"{p.grad} vs {p_copy.grad}"


if __name__ == '__main__':
    test_base_param_hook()

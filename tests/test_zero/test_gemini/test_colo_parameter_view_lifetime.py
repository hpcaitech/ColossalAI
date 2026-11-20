import torch

from colossalai.tensor.colo_parameter import ColoParameter
from colossalai.tensor.param_op_hook import ColoParamOpHook, ColoParamOpHookManager
from colossalai.utils import free_storage


class ReleaseParameterStorageHook(ColoParamOpHook):
    def pre_forward(self, params):
        pass

    def post_forward(self, params):
        for param in params:
            free_storage(param)

    def pre_backward(self, params):
        pass

    def post_backward(self, params):
        pass


def test_indexed_3d_parameter_survives_storage_release():
    param = ColoParameter(torch.randn(4, 3, 2, requires_grad=True))
    hook = ReleaseParameterStorageHook()

    with ColoParamOpHookManager.use_hooks(hook):
        expert = param[0]
        assert expert.storage().size() == 3 * 2
        expert.square().sum().backward()

    assert param.grad is not None
    assert param.grad.shape == param.shape

from typing import List

from torch._tensor import Tensor

from colossalai.tensor.param_op_hook import ColoParamOpHook

_ALL_GATHER_HANDLE = "_all_gather_handle"


def wait_all_gather_handle(p):
    if hasattr(p, _ALL_GATHER_HANDLE):
        handle = getattr(p, _ALL_GATHER_HANDLE)
        handle.wait()
        delattr(p, _ALL_GATHER_HANDLE)


def set_all_gather_handle(p, handle):
    setattr(p, _ALL_GATHER_HANDLE, handle)


class ZeroOpHook(ColoParamOpHook):
    def pre_forward(self, params: List[Tensor]) -> None:
        for p in params:
            wait_all_gather_handle(p)

    def post_forward(self, params: List[Tensor]) -> None:
        pass

    def pre_backward(self, params: List[Tensor]) -> None:
        pass

    def post_backward(self, params: List[Tensor]) -> None:
        pass

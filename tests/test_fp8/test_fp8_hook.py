import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from colossalai.accelerator import get_accelerator
from colossalai.quantization.fp8 import linear_fp8, linear_fp8_deep_gemm
from colossalai.quantization.fp8_hook import FP8DeepGemmHook, FP8Hook
from colossalai.tensor.colo_parameter import ColoParameter
from colossalai.tensor.param_op_hook import ColoParamOpHookManager
from colossalai.utils import get_current_device

REPLACED = False
TRIGGERED = False


def new_linear_fp8(x, w, bias=None):
    global TRIGGERED
    TRIGGERED = True
    return linear_fp8(x, w, bias)


def new_deepgemm_fp8_gemm(lhs, rhs, out=None):
    global TRIGGERED
    TRIGGERED = True
    return linear_fp8_deep_gemm(lhs, rhs, out)


class FP8TestHook(FP8Hook):
    def rewrite_op(self, func):
        func = super().rewrite_op(func)
        if func is linear_fp8:
            global REPLACED
            REPLACED = True
            return new_linear_fp8
        return func


class DeepGemmTestHook(FP8DeepGemmHook):
    def rewrite_op(self, func):
        func = super().rewrite_op(func)
        if func is linear_fp8_deep_gemm:
            global REPLACED
            REPLACED = True
            return new_deepgemm_fp8_gemm
        return func


D_IN, D_OUT = 128, 128
B, S = 2, 64
DTYPE = torch.bfloat16


@pytest.mark.skipif(get_accelerator().get_device_capability()[0] < 9, reason="Test requires device capability >= 9.0")
def test_fp8_hook():
    global REPLACED, TRIGGERED
    REPLACED = False
    TRIGGERED = False
    # create tensors
    w = nn.Parameter(torch.rand(D_OUT, D_IN, device=get_current_device(), dtype=DTYPE))
    x = torch.rand(B, S, D_IN, device=get_current_device(), dtype=DTYPE, requires_grad=True)
    w.__class__ = ColoParameter
    w.__init__(w, requires_grad=True)
    hook = FP8TestHook()
    with ColoParamOpHookManager.use_hooks(hook):
        o = F.linear(x, w)
    assert o.shape == (B, S, D_OUT)
    assert REPLACED
    assert TRIGGERED


@pytest.mark.skipif(get_accelerator().get_device_capability()[0] < 9, reason="Test requires device capability >= 9.0")
def test_fp8_deep_gemm_hook():
    global REPLACED, TRIGGERED
    REPLACED = False
    TRIGGERED = False
    # create tensors
    w = nn.Parameter(torch.rand(D_OUT, D_IN, device=get_current_device(), dtype=DTYPE))
    x = torch.rand(S, D_IN, device=get_current_device(), dtype=DTYPE, requires_grad=True)
    w.__class__ = ColoParameter
    w.__init__(w, requires_grad=True)
    hook = DeepGemmTestHook()
    with ColoParamOpHookManager.use_hooks(hook):
        o = F.linear(x, w)
    assert o.shape == (S, D_OUT)
    assert REPLACED
    assert TRIGGERED

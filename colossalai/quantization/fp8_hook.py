import torch.nn.functional as F

from colossalai.quantization.fp8 import linear_fp8, linear_fp8_deep_gemm
from colossalai.tensor.param_op_hook import ColoParamOpHook


class FP8Hook(ColoParamOpHook):
    def pre_forward(self, params) -> None:
        pass

    def post_forward(self, params) -> None:
        pass

    def pre_backward(self, params) -> None:
        pass

    def post_backward(self, params) -> None:
        pass

    def rewrite_op(self, func):
        if func is F.linear:
            return linear_fp8
        return func


class FP8DeepGemmHook(ColoParamOpHook):

    def pre_forward(self, params) -> None:
        pass

    def post_forward(self, params) -> None:
        pass

    def pre_backward(self, params) -> None:
        pass

    def post_backward(self, params) -> None:
        pass

    def rewrite_op(self, func):
        if func is F.linear:
            return linear_fp8_deep_gemm
        return func

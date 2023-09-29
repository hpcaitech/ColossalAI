# adopted from https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/jit_handles.py
# ideas from https://pastebin.com/AkvAyJBw
# and https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505

import operator
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum, auto
from functools import partial, reduce
from numbers import Number
from typing import Any, Callable, List, Union

import torch
from packaging import version
from torch.utils._pytree import tree_map

from .meta_tensor import MetaTensor

aten = torch.ops.aten


class Phase(Enum):
    FWD = auto()
    BWD = auto()


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


def _format_flops(flop):
    K = 1e3
    M = 1e6
    B = 1e9
    T = 1e12
    if flop < K:
        return f"{flop:.2f}"
    elif flop < M:
        return f"{flop / K:.2f}K"
    elif flop < B:
        return f"{flop / M:.2f}M"
    elif flop < T:
        return f"{flop / B:.2f}B"
    else:
        return f"{flop / T:.2f}T"


def flop_count(module: Union[torch.nn.Module, Callable] = None, *args, verbose: bool = False, **kwargs) -> Number:
    """
    Count the number of floating point operations in a model.
    Ideas from https://pastebin.com/AkvAyJBw.
    Args:
        module (torch.nn.Module): A PyTorch model.
        *args: Input arguments to the model.
        verbose (bool): If True, print the number of flops for each module.
        **kwargs: Input keyword arguments to the model.
    Returns:
        Number: The total number of floating point operations (FWD + BWD).
    """
    maybe_inplace = (
        getattr(module, "inplace", False)
        or kwargs.get("inplace", False)
        or getattr(module, "__name__", None) in ("add_", "mul_", "div_", "sub_")
    )

    class DummyModule(torch.nn.Module):
        def __init__(self, func):
            super().__init__()
            self.func = func
            self.__name__ = func.__name__

        def forward(self, *args, **kwargs):
            return self.func(*args, **kwargs)

    total_flop_count = {Phase.FWD: 0, Phase.BWD: 0}
    flop_counts = defaultdict(lambda: defaultdict(int))
    parents = ["Global"]
    module = module if isinstance(module, torch.nn.Module) else DummyModule(module)

    class FlopTensor(MetaTensor):
        _tensor: torch.Tensor

        def __repr__(self):
            name = "FlopParameter" if getattr(self, "_is_param", False) else "FlopTensor"
            if self.grad_fn:
                return f"{name}(..., size={tuple(self.shape)}, device='{self.device}', dtype={self.dtype}, grad_fn={self.grad_fn})"
            return f"{name}(..., size={tuple(self.shape)}, device='{self.device}', dtype={self.dtype})"

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            # no_dispatch is only needed if you use enable_python_mode.
            # It prevents infinite recursion.
            rs = super().__torch_dispatch__(func, types, args, kwargs)

            outs = normalize_tuple(rs)

            if func in flop_mapping:
                nonlocal flop_counts, total_flop_count
                flop_count = flop_mapping[func](args, outs)
                for par in parents:
                    flop_counts[par][func.__name__] += flop_count
                total_flop_count[cur_phase] += flop_count

            def wrap(x):
                if isinstance(x, MetaTensor):
                    x = FlopTensor(x)
                return x

            rs = tree_map(wrap, rs)

            return rs

    def is_autogradable(x):
        return isinstance(x, torch.Tensor) and x.is_floating_point()

    def create_backwards_push(name):
        class PushState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                nonlocal parents
                parents.append(name)
                return grad_outs

        return PushState.apply

    def create_backwards_pop(name):
        class PopState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                nonlocal parents
                assert parents[-1] == name
                parents.pop()
                return grad_outs

        return PopState.apply

    def enter_module(name):
        def f(module, inputs):
            nonlocal parents
            parents.append(name)
            inputs = normalize_tuple(inputs)
            out = create_backwards_pop(name)(*inputs)
            return out

        return f

    def exit_module(name):
        def f(module, inputs, outputs):
            nonlocal parents
            assert parents[-1] == name
            parents.pop()
            outputs = normalize_tuple(outputs)
            return create_backwards_push(name)(*outputs)

        return f

    @contextmanager
    def instrument_module(mod):
        registered = []
        for name, module in dict(mod.named_children()).items():
            registered.append(module.register_forward_pre_hook(enter_module(name)))
            registered.append(module.register_forward_hook(exit_module(name)))
        yield
        for handle in registered:
            handle.remove()

    def display_flops():
        for mod in flop_counts.keys():
            print(f"Module: ", mod)
            for k, v in flop_counts[mod].items():
                print("\t", k, _format_flops(v))
            print()

    def detach_variables(r):
        if isinstance(r, torch.Tensor):
            requires_grad = r.requires_grad
            r = r.detach()
            r.requires_grad = requires_grad
        return r

    def wrap(r):
        if isinstance(r, torch.Tensor):
            data_ptr_fn = getattr(r, "_tensor", r).data_ptr
            r = FlopTensor(detach_variables(r))
            if maybe_inplace:
                r = r + 0
            r._tensor.data_ptr = data_ptr_fn
        return r

    with instrument_module(module):
        cur_phase = Phase.FWD
        rst = module(*tree_map(wrap, args), **tree_map(wrap, kwargs))
        rst = tuple(r for r in normalize_tuple(rst) if is_autogradable(r) and r.requires_grad)
        cur_phase = Phase.BWD

        if rst:
            grad = [torch.zeros_like(t) for t in rst]
            torch.autograd.backward(
                rst,
                grad,
            )

    if verbose:
        display_flops()

    return total_flop_count[Phase.FWD], total_flop_count[Phase.BWD]


def matmul_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for matmul.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    input_shapes = [v.shape for v in inputs]
    assert len(input_shapes) == 2, input_shapes

    # There are three cases: 1) gemm, 2) gemv, 3) dot
    if all(len(shape) == 2 for shape in input_shapes):
        # gemm
        assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
    elif all(len(shape) == 1 for shape in input_shapes):
        # dot
        assert input_shapes[0][0] == input_shapes[1][0], input_shapes

        # expand shape
        input_shapes[0] = torch.Size([1, input_shapes[0][0]])
        input_shapes[1] = torch.Size([input_shapes[1][0], 1])
    else:
        # gemv
        if len(input_shapes[0]) == 1:
            assert input_shapes[0][0] == input_shapes[1][-2], input_shapes
            input_shapes.reverse()
        else:
            assert input_shapes[1][0] == input_shapes[0][-1], input_shapes

        # expand the shape of the vector to [batch size, 1]
        input_shapes[-1] = torch.Size([input_shapes[-1][-1], 1])
    flops = reduce(operator.mul, input_shapes[0]) * input_shapes[-1][-1]
    return flops


def addmm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for fully connected layers.
    """
    # Count flop for nn.Linear
    # inputs is a list of length 3.
    input_shapes = [v.shape for v in inputs[1:3]]
    # input_shapes[0]: [batch size, input feature dimension]
    # input_shapes[1]: [input feature dimension, output feature dimension]
    assert len(input_shapes[0]) == 2, input_shapes[0]
    assert len(input_shapes[1]) == 2, input_shapes[1]
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    flops = batch_size * input_dim * output_dim
    return flops


def linear_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the aten::linear operator.
    """
    # Inputs is a list of length 3; unlike aten::addmm, it is the first
    # two elements that are relevant.
    input_shapes = [v.shape for v in inputs[0:2]]
    # input_shapes[0]: [dim0, dim1, ..., input_feature_dim]
    # input_shapes[1]: [output_feature_dim, input_feature_dim]
    assert input_shapes[0][-1] == input_shapes[1][-1]
    flops = reduce(operator.mul, input_shapes[0]) * input_shapes[1][0]
    return flops


def bmm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the bmm operation.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    assert len(inputs) == 2, len(inputs)
    input_shapes = [v.shape for v in inputs]
    n, c, t = input_shapes[0]
    d = input_shapes[-1][-1]
    flops = n * c * t * d
    return flops


def conv_flop_count(
    x_shape: List[int],
    w_shape: List[int],
    out_shape: List[int],
    transposed: bool = False,
) -> Number:
    """
    Count flops for convolution. Note only multiplication is
    counted. Computation for addition and bias is ignored.
    Flops for a transposed convolution are calculated as
    flops = (x_shape[2:] * prod(w_shape) * batch_size).
    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed
    Returns:
        int: the number of flops
    """
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    flops = batch_size * reduce(operator.mul, w_shape) * reduce(operator.mul, conv_shape)
    return flops


def conv_flop_jit(inputs: List[Any], outputs: List[Any]):
    """
    Count flops for convolution.
    """
    x, w = inputs[:2]
    x_shape, w_shape, out_shape = (x.shape, w.shape, outputs[0].shape)
    transposed = inputs[6]

    return conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed)


def transpose_shape(shape):
    return [shape[1], shape[0]] + list(shape[2:])


def conv_backward_flop_jit(inputs: List[Any], outputs: List[Any]):
    grad_out_shape, x_shape, w_shape = [i.shape for i in inputs[:3]]
    output_mask = inputs[-1]
    fwd_transposed = inputs[7]
    flop_count = 0

    if output_mask[0]:
        grad_input_shape = outputs[0].shape
        flop_count += conv_flop_count(grad_out_shape, w_shape, grad_input_shape, not fwd_transposed)
    if output_mask[1]:
        grad_weight_shape = outputs[1].shape
        flop_count += conv_flop_count(transpose_shape(x_shape), grad_out_shape, grad_weight_shape, fwd_transposed)

    return flop_count


def norm_flop_counter(affine_arg_index: int, input_arg_index: int) -> Callable:
    """
    Args:
        affine_arg_index: index of the affine argument in inputs
    """

    def norm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
        """
        Count flops for norm layers.
        """
        # Inputs[0] contains the shape of the input.
        input_shape = inputs[input_arg_index].shape

        has_affine = (
            inputs[affine_arg_index].shape is not None
            if hasattr(inputs[affine_arg_index], "shape")
            else inputs[affine_arg_index]
        )
        assert 2 <= len(input_shape) <= 5, input_shape
        # 5 is just a rough estimate
        flop = reduce(operator.mul, input_shape) * (5 if has_affine else 4)
        return flop

    return norm_flop_jit


def batchnorm_flop_jit(inputs: List[Any], outputs: List[Any], training: bool = None) -> Number:
    if training is None:
        training = inputs[-3]
    assert isinstance(training, bool), "Signature of aten::batch_norm has changed!"
    if training:
        return norm_flop_counter(1, 0)(inputs, outputs)  # pyre-ignore
    has_affine = inputs[1].shape is not None
    input_shape = reduce(operator.mul, inputs[0].shape)
    return input_shape * (2 if has_affine else 1)


def ewise_flop_counter(input_scale: float = 1, output_scale: float = 0) -> Callable:
    """
    Count flops by
        input_tensor.numel() * input_scale + output_tensor.numel() * output_scale
    Args:
        input_scale: scale of the input tensor (first argument)
        output_scale: scale of the output tensor (first element in outputs)
    """

    def ewise_flop(inputs: List[Any], outputs: List[Any]) -> Number:
        ret = 0
        if input_scale != 0:
            shape = inputs[0].shape
            ret += input_scale * reduce(operator.mul, shape) if shape else 0
        if output_scale != 0:
            shape = outputs[0].shape
            ret += output_scale * reduce(operator.mul, shape) if shape else 0
        return ret

    return ewise_flop


def zero_flop_jit(*args):
    """
    Count flops for zero flop layers.
    """
    return 0


if version.parse(torch.__version__) >= version.parse("1.12.0"):
    flop_mapping = {
        # gemm
        aten.mm.default: matmul_flop_jit,
        aten.matmul.default: matmul_flop_jit,
        aten.addmm.default: addmm_flop_jit,
        aten.bmm.default: bmm_flop_jit,
        # convolution
        aten.convolution.default: conv_flop_jit,
        aten._convolution.default: conv_flop_jit,
        aten.convolution_backward.default: conv_backward_flop_jit,
        # normalization
        aten.native_batch_norm.default: batchnorm_flop_jit,
        aten.native_batch_norm_backward.default: batchnorm_flop_jit,
        aten.cudnn_batch_norm.default: batchnorm_flop_jit,
        aten.cudnn_batch_norm_backward.default: partial(batchnorm_flop_jit, training=True),
        aten.native_layer_norm.default: norm_flop_counter(2, 0),
        aten.native_layer_norm_backward.default: norm_flop_counter(2, 0),
        # pooling
        aten.avg_pool1d.default: ewise_flop_counter(1, 0),
        aten.avg_pool2d.default: ewise_flop_counter(1, 0),
        aten.avg_pool2d_backward.default: ewise_flop_counter(0, 1),
        aten.avg_pool3d.default: ewise_flop_counter(1, 0),
        aten.avg_pool3d_backward.default: ewise_flop_counter(0, 1),
        aten.max_pool1d.default: ewise_flop_counter(1, 0),
        aten.max_pool2d.default: ewise_flop_counter(1, 0),
        aten.max_pool3d.default: ewise_flop_counter(1, 0),
        aten.max_pool1d_with_indices.default: ewise_flop_counter(1, 0),
        aten.max_pool2d_with_indices.default: ewise_flop_counter(1, 0),
        aten.max_pool2d_with_indices_backward.default: ewise_flop_counter(0, 1),
        aten.max_pool3d_with_indices.default: ewise_flop_counter(1, 0),
        aten.max_pool3d_with_indices_backward.default: ewise_flop_counter(0, 1),
        aten._adaptive_avg_pool2d.default: ewise_flop_counter(1, 0),
        aten._adaptive_avg_pool2d_backward.default: ewise_flop_counter(0, 1),
        aten._adaptive_avg_pool3d.default: ewise_flop_counter(1, 0),
        aten._adaptive_avg_pool3d_backward.default: ewise_flop_counter(0, 1),
        aten.embedding_dense_backward.default: ewise_flop_counter(0, 1),
        aten.embedding.default: ewise_flop_counter(1, 0),
    }

    ewise_flop_aten = [
        # basic op
        aten.add.Tensor,
        aten.add_.Tensor,
        aten.div.Tensor,
        aten.div_.Tensor,
        aten.div.Scalar,
        aten.div_.Scalar,
        aten.mul.Tensor,
        aten.mul.Scalar,
        aten.mul_.Tensor,
        aten.neg.default,
        aten.pow.Tensor_Scalar,
        aten.rsub.Scalar,
        aten.sum.default,
        aten.sum.dim_IntList,
        aten.mean.dim,
        # activation op
        aten.hardswish.default,
        aten.hardswish_.default,
        aten.hardswish_backward.default,
        aten.hardtanh.default,
        aten.hardtanh_.default,
        aten.hardtanh_backward.default,
        aten.hardsigmoid_backward.default,
        aten.hardsigmoid.default,
        aten.gelu.default,
        aten.gelu_backward.default,
        aten.silu.default,
        aten.silu_.default,
        aten.silu_backward.default,
        aten.sigmoid.default,
        aten.sigmoid_backward.default,
        aten._softmax.default,
        aten._softmax_backward_data.default,
        aten.relu_.default,
        aten.relu.default,
        aten.tanh.default,
        aten.tanh_backward.default,
        aten.threshold_backward.default,
        # dropout
        aten.native_dropout.default,
        aten.native_dropout_backward.default,
        # distribution
        aten.bernoulli_.float,
        # where
        aten.where.self,
    ]
    for op in ewise_flop_aten:
        flop_mapping[op] = ewise_flop_counter(1, 0)

    # fix-me: this will be removed in future
    zero_flop_aten = [
        aten.as_strided.default,
        aten.as_strided_.default,
        aten.cat.default,
        aten.clone.default,
        aten.copy_.default,
        aten.detach.default,
        aten.expand.default,
        aten.empty_like.default,
        aten.new_empty.default,
        aten.new_empty_strided.default,
        aten.ones_like.default,
        aten._reshape_alias.default,
        aten.select.int,
        aten.select_backward.default,
        aten.squeeze.dim,
        aten.slice.Tensor,
        aten.slice_backward.default,
        aten.split.Tensor,
        aten.permute.default,
        aten.t.default,
        aten.transpose.int,
        aten._to_copy.default,
        aten.unsqueeze.default,
        aten.unbind.int,
        aten._unsafe_view.default,
        aten.view.default,
        aten.zero_.default,
        aten.zeros_like.default,
    ]

    for op in zero_flop_aten:
        flop_mapping[op] = zero_flop_jit
else:
    flop_mapping = {}
    elementwise_flop_aten = {}
    zero_flop_aten = {}

# meta patch from https://github.com/pytorch/pytorch/blob/master/torch/_meta_registrations.py
# should be activated for PyTorch version 1.12.0 and below
# refer to https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml
# for more meta_registrations

from typing import Callable, List, Optional, Tuple, Union

import torch
from torch.utils._pytree import tree_map

aten = torch.ops.aten

meta_lib = torch.library.Library("aten", "IMPL", "Meta")

meta_table = {}


def register_meta(op, register_dispatcher=True):

    def wrapper(f):

        def add_func(op):
            meta_table[op] = f
            if register_dispatcher:
                name = (op.__name__ if op._overloadname != "default" else op.overloadpacket.__name__)
                try:
                    meta_lib.impl(name, f)
                except:
                    pass

        tree_map(add_func, op)
        return f

    return wrapper


# ============================== Convolutions ======================================
# https://github.com/pytorch/pytorch/pull/79834
@register_meta(aten.convolution.default)
def meta_conv(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    is_transposed: bool,
    output_padding: List[int],
    groups: int,
):

    def _formula(ln: int, p: int, d: int, k: int, s: int) -> int:
        """
        Formula to apply to calculate the length of some dimension of the output
        See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        Args:
            ln: length of the dimension
            p: padding in that dim
            d: dilation in that dim
            k: kernel size in that dim
            s: stride in that dim
        Returns:
            The output length
        """
        return (ln + 2 * p - d * (k - 1) - 1) // s + 1

    def _formula_transposed(ln: int, p: int, d: int, k: int, s: int, op: int) -> int:
        """
        Formula to apply to calculate the length of some dimension of the output
        if transposed convolution is used.
        See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        Args:
            ln: length of the dimension
            p: padding in that dim
            d: dilation in that dim
            k: kernel size in that dim
            s: stride in that dim
            op: output padding in that dim
        Returns:
            The output length
        """
        return (ln - 1) * s - 2 * p + d * (k - 1) + op + 1

    def calc_conv_nd_return_shape(
        dims: torch.Size,
        kernel_size: torch.Size,
        stride: Union[List[int], int],
        padding: Union[List[int], int],
        dilation: Union[List[int], int],
        output_padding: Optional[Union[List[int], int]] = None,
    ):
        ret_shape = []
        if isinstance(stride, int):
            stride = [stride] * len(dims)
        elif len(stride) == 1:
            stride = [stride[0]] * len(dims)

        if isinstance(padding, int):
            padding = [padding] * len(dims)
        elif len(padding) == 1:
            padding = [padding[0]] * len(dims)

        if isinstance(dilation, int):
            dilation = [dilation] * len(dims)
        elif len(dilation) == 1:
            dilation = [dilation[0]] * len(dims)

        output_padding_list: Optional[List[int]] = None
        if output_padding:
            if isinstance(output_padding, int):
                output_padding_list = [output_padding] * len(dims)
            elif len(output_padding) == 1:
                output_padding_list = [output_padding[0]] * len(dims)
            else:
                output_padding_list = output_padding

        for i in range(len(dims)):
            # If output_padding is present, we are dealing with a transposed convolution
            if output_padding_list:
                ret_shape.append(
                    _formula_transposed(
                        dims[i],
                        padding[i],
                        dilation[i],
                        kernel_size[i],
                        stride[i],
                        output_padding_list[i],
                    ))
            else:
                ret_shape.append(_formula(dims[i], padding[i], dilation[i], kernel_size[i], stride[i]))
        return ret_shape

    def pick_memory_format():
        if input_tensor.is_contiguous(memory_format=torch.channels_last):
            return torch.channels_last
        elif input_tensor.is_contiguous(memory_format=torch.contiguous_format):
            return torch.contiguous_format
        elif input_tensor.is_contiguous(memory_format=torch.preserve_format):
            return torch.preserve_format

    kernel_size = weight.shape[2:]
    dims = input_tensor.shape[2:]
    if is_transposed:
        out_channels = groups * weight.shape[1]

        shape_out = calc_conv_nd_return_shape(
            dims,
            kernel_size,
            stride,
            padding,
            dilation,
            output_padding,
        )

    else:
        out_channels = weight.shape[0]
        if weight.shape[1] != input_tensor.shape[1] / groups:
            raise RuntimeError("Invalid channel dimensions")
        shape_out = calc_conv_nd_return_shape(dims, kernel_size, stride, padding, dilation)
    out = input_tensor.new_empty((input_tensor.shape[0], out_channels, *shape_out))
    mem_fmt = pick_memory_format()
    out = out.to(memory_format=mem_fmt)    # type: ignore[call-overload]
    return out


@register_meta(aten._convolution.default)
def meta_conv_1(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: List[int],
                padding: List[int], dilation: List[int], is_transposed: bool, output_padding: List[int], groups: int,
                *extra_args):
    out = meta_conv(input_tensor, weight, bias, stride, padding, dilation, is_transposed, output_padding, groups)
    return out


@register_meta(aten.convolution_backward.default)
def meta_conv_backward(grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, bias_sizes, stride,
                       padding, dilation, transposed, output_padding, groups, output_mask):
    return torch.empty_like(input), torch.empty_like(weight), torch.empty((bias_sizes), device='meta')


# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/AdaptiveAveragePooling.cpp
@register_meta(aten._adaptive_avg_pool2d_backward.default)
def meta_adaptive_avg_pool2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
):
    grad_input = torch.empty_like(input)
    return grad_input


# ================================ RNN =============================================
# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cudnn/RNN.cpp
@register_meta(aten._cudnn_rnn.default)
def meta_cuda_rnn(
    input,
    weight,
    weight_stride0,
    weight_buf,
    hx,
    cx,
    mode,
    hidden_size,
    proj_size,
    num_layers,
    batch_first,
    dropout,
    train,
    bidirectional,
    batch_sizes,
    dropout_state,
):

    is_input_packed = len(batch_sizes) != 0
    if is_input_packed:
        seq_length = len(batch_sizes)
        mini_batch = batch_sizes[0]
        batch_sizes_sum = input.shape[0]
    else:
        seq_length = input.shape[1] if batch_first else input.shape[0]
        mini_batch = input.shape[0] if batch_first else input.shape[1]
        batch_sizes_sum = -1

    num_directions = 2 if bidirectional else 1
    out_size = proj_size if proj_size != 0 else hidden_size
    if is_input_packed:
        out_shape = [batch_sizes_sum, out_size * num_directions]
    else:
        out_shape = ([mini_batch, seq_length, out_size *
                      num_directions] if batch_first else [seq_length, mini_batch, out_size * num_directions])
    output = input.new_empty(out_shape)

    cell_shape = [num_layers * num_directions, mini_batch, hidden_size]
    cy = torch.empty(0) if cx is None else cx.new_empty(cell_shape)

    hy = hx.new_empty([num_layers * num_directions, mini_batch, out_size])

    # TODO: Query cudnnGetRNNTrainingReserveSize (expose to python)
    reserve_shape = 0 if train else 0
    reserve = input.new_empty(reserve_shape, dtype=torch.uint8)

    return output, hy, cy, reserve, weight_buf


# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cudnn/RNN.cpp
@register_meta(aten._cudnn_rnn_backward.default)
def meta_cudnn_rnn_backward(input: torch.Tensor,
                            weight: torch.Tensor,
                            weight_stride0: int,
                            hx: torch.Tensor,
                            cx: Optional[torch.Tensor] = None,
                            *args,
                            **kwargs):
    print(input, weight, hx, cx)
    grad_input = torch.empty_like(input)
    grad_weight = torch.empty_like(weight)
    grad_hx = torch.empty_like(hx)
    grad_cx = torch.empty_like(cx) if cx is not None else torch.empty((), device='meta')
    return grad_input, grad_weight, grad_hx, grad_cx


# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Activation.cpp
# ============================== Activations =======================================
@register_meta(aten.relu.default)
def meta_relu(input: torch.Tensor):
    return torch.empty_like(input)


@register_meta(aten.prelu.default)
def meta_prelu(input: torch.Tensor, weight: torch.Tensor):
    return torch.empty_like(input)


@register_meta(aten.hardswish.default)
def meta_hardswish(input: torch.Tensor):
    return torch.empty_like(input)


@register_meta(aten.hardtanh.default)
def meta_hardtanh(input: torch.Tensor, min, max):
    return torch.empty_like(input)


@register_meta(aten.hardswish_backward.default)
def meta_hardswish_backward(grad_out: torch.Tensor, input: torch.Tensor):
    grad_in = torch.empty_like(input)
    return grad_in


@register_meta(aten.hardtanh_backward.default)
def meta_hardtanh_backward(grad_out: torch.Tensor, input: torch.Tensor, min_val: int, max_val: int):
    grad_in = torch.empty_like(input)
    return grad_in


# ============================== Normalization =====================================
# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cudnn/BatchNorm.cpp
@register_meta(aten.native_batch_norm.default)
def meta_bn(input: torch.Tensor, weight, bias, running_mean, running_var, training, momentum, eps):
    n_input = input.size(1)

    output = torch.empty_like(input)
    running_mean = torch.empty((n_input), device='meta')
    running_var = torch.empty((n_input), device='meta')
    return output, running_mean, running_var


# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cudnn/BatchNorm.cpp
@register_meta(aten.native_batch_norm_backward.default)
def meta_bn_backward(dY: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, running_mean, running_var, save_mean,
                     save_invstd, train, eps, output_mask):
    dX = torch.empty_like(input)
    dgamma = torch.empty_like(weight)
    dbeta = torch.empty_like(weight)
    return dX, dgamma, dbeta


# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cudnn/BatchNorm.cpp
@register_meta(aten.cudnn_batch_norm.default)
def meta_cudnn_bn(input: torch.Tensor, weight, bias, running_mean, running_var, training, momentum, eps):
    n_input = input.size(1)

    output = torch.empty_like(input)
    running_mean = torch.empty((n_input), device='meta')
    running_var = torch.empty((n_input), device='meta')
    reserve = torch.empty((0), dtype=torch.uint8, device='meta')
    return output, running_mean, running_var, reserve


# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cudnn/BatchNorm.cpp
# NB: CuDNN only implements the backward algorithm for batchnorm
# in training mode (evaluation mode batchnorm has a different algorithm),
# which is why this doesn't accept a 'training' parameter.
@register_meta(aten.cudnn_batch_norm_backward.default)
def meta_cudnn_bn_backward(dY: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, running_mean, running_var,
                           save_mean, save_invstd, eps, reserve):
    dX = torch.empty_like(input)
    dgamma = torch.empty_like(weight)
    dbeta = torch.empty_like(weight)
    return dX, dgamma, dbeta


# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/layer_norm.cpp
@register_meta(aten.native_layer_norm.default)
def meta_ln(input: torch.Tensor, normalized_shape, weight, bias, eps):
    bs = input.size(0)
    n_input = input.size(1)

    output = torch.empty_like(input)
    running_mean = torch.empty((bs, n_input, 1), device='meta')
    running_var = torch.empty((bs, n_input, 1), device='meta')
    return output, running_mean, running_var


# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/layer_norm.cpp
@register_meta(aten.native_layer_norm_backward.default)
def meta_ln_backward(dY: torch.Tensor, input: torch.Tensor, normalized_shape, mean, rstd, weight, bias,
                     grad_input_mask):
    dX = torch.empty_like(input)
    dgamma = torch.empty_like(weight)
    dbeta = torch.empty_like(bias)
    return dX, dgamma, dbeta


# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/group_norm.cpp
@register_meta(aten.native_group_norm_backward.default)
def meta_gn_backward(dY: torch.Tensor, input: torch.Tensor, mean, rstd, gamma, N, C, HxW, group, grad_input_mask):
    dX = torch.empty_like(input)
    dgamma = torch.empty_like(gamma)
    dbeta = torch.empty_like(gamma)
    return dX, dgamma, dbeta


# ================================== Misc ==========================================
# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml
@register_meta(aten.roll.default)
def meta_roll(input: torch.Tensor, shifts, dims):
    return input


# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Scalar.cpp
@register_meta(aten._local_scalar_dense.default)
def meta_local_scalar_dense(self: torch.Tensor):
    return 0


# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorCompare.cpp
@register_meta(aten.where.self)
def meta_where_self(condition: torch.Tensor, self: torch.Tensor, other: torch.Tensor):
    result_type = torch.result_type(self, other)
    return torch.empty_like(self, dtype=result_type)


@register_meta(aten.index.Tensor)
def meta_index_Tensor(self, indices):
    assert indices, "at least one index must be provided"
    # aten::index is the internal advanced indexing implementation
    # checkIndexTensorTypes and expandTensors
    result: List[Optional[torch.Tensor]] = []
    for i, index in enumerate(indices):
        if index is not None:
            assert index.dtype in [torch.long, torch.int8, torch.bool],\
                "tensors used as indices must be long, byte or bool tensors"
            if index.dtype in [torch.int8, torch.bool]:
                nonzero = index.nonzero()
                k = len(result)
                assert k + index.ndim <= self.ndim, f"too many indices for tensor of dimension {self.ndim}"
                for j in range(index.ndim):
                    assert index.shape[j] == self.shape[
                        k +
                        j], f"The shape of the mask {index.shape} at index {i} does not match the shape of the indexed tensor {self.shape} at index {k + j}"
                    result.append(nonzero.select(1, j))
            else:
                result.append(index)
        else:
            result.append(index)
    indices = result
    assert len(indices) <= self.ndim, f"too many indices for tensor of dimension {self.ndim} (got {len(indices)})"
    # expand_outplace
    import torch._refs as refs

    indices = list(refs._maybe_broadcast(*indices))
    # add missing null tensors
    while len(indices) < self.ndim:
        indices.append(None)

    # hasContiguousSubspace
    #   true if all non-null tensors are adjacent
    # See:
    # https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
    # https://stackoverflow.com/questions/53841497/why-does-numpy-mixed-basic-advanced-indexing-depend-on-slice-adjacency
    state = 0
    has_contiguous_subspace = False
    for index in indices:
        if state == 0:
            if index is not None:
                state = 1
        elif state == 1:
            if index is None:
                state = 2
        else:
            if index is not None:
                break
    else:
        has_contiguous_subspace = True

    # transposeToFront
    # This is the logic that causes the newly inserted dimensions to show up
    # at the beginning of the tensor, if they're not contiguous
    if not has_contiguous_subspace:
        dims = []
        transposed_indices = []
        for i, index in enumerate(indices):
            if index is not None:
                dims.append(i)
                transposed_indices.append(index)
        for i, index in enumerate(indices):
            if index is None:
                dims.append(i)
                transposed_indices.append(index)
        self = self.permute(dims)
        indices = transposed_indices

    # AdvancedIndex::AdvancedIndex
    # Now we can assume the indices have contiguous subspace
    # This is simplified from AdvancedIndex which goes to more effort
    # to put the input and indices in a form so that TensorIterator can
    # take them.  If we write a ref for this, probably that logic should
    # get implemented
    before_shape: List[int] = []
    after_shape: List[int] = []
    replacement_shape: List[int] = []
    for dim, index in enumerate(indices):
        if index is None:
            if replacement_shape:
                after_shape.append(self.shape[dim])
            else:
                before_shape.append(self.shape[dim])
        else:
            replacement_shape = list(index.shape)
    return self.new_empty(before_shape + replacement_shape + after_shape)


# ============================== Embedding =========================================
# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Embedding.cpp
@register_meta(aten.embedding_dense_backward.default)
def meta_embedding_dense_backward(grad_output: torch.Tensor, indices: torch.Tensor, num_weights, padding_idx,
                                  scale_grad_by_freq):
    return torch.empty((num_weights, grad_output.size(-1)),
                       dtype=grad_output.dtype,
                       device=grad_output.device,
                       layout=grad_output.layout)


# ============================== Dropout ===========================================
# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Dropout.cpp
@register_meta(aten.native_dropout.default)
def meta_native_dropout_default(input: torch.Tensor, p: float, train: bool = False):
    # notice that mask is bool
    output = torch.empty_like(input)
    mask = torch.empty_like(input, dtype=torch.bool)
    return output, mask


# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Dropout.cpp
@register_meta(aten.native_dropout_backward.default)
def meta_native_dropout_backward_default(grad: torch.Tensor, mask: torch.Tensor, scale: float):
    return torch.empty_like(grad)

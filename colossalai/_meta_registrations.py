# meta patch from https://github.com/pytorch/pytorch/blob/master/torch/_meta_registrations.py
# should be activated for PyTorch version 1.12.0 and below

from typing import List, Optional, Tuple, Union
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


@register_meta(aten.convolution_backward.default)
def meta_conv_backward(grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, bias_sizes, stride,
                       padding, dilation, transposed, output_padding, groups, output_mask):
    return torch.empty_like(input), torch.empty_like(weight), torch.empty((bias_sizes), device='meta')


@register_meta(aten.relu.default)
def meta_relu(input: torch.Tensor):
    return torch.empty_like(input)


@register_meta(aten.hardswish.default)
def meta_hardswish(input: torch.Tensor):
    return torch.empty_like(input)


@register_meta(aten.hardswish_backward.default)
def meta_hardswish_backward(grad_out: torch.Tensor, input: torch.Tensor):
    grad_in = torch.empty_like(input)
    return grad_in


@register_meta(aten.roll.default)
def meta_roll(input: torch.Tensor, shifts, dims):
    return torch.empty_like(input)


@register_meta(aten.native_batch_norm.default)
def meta_bn(input: torch.Tensor, weight, bias, running_mean, running_var, training, momentum, eps):
    n_input = input.size(1)

    output = torch.empty_like(input)
    running_mean = torch.empty((n_input), device='meta')
    running_var = torch.empty((n_input), device='meta')
    return output, running_mean, running_var


@register_meta(aten.native_batch_norm_backward.default)
def meta_bn_backward(dY: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, running_mean, running_var, save_mean,
                     save_invstd, train, eps, output_mask):
    dX = torch.empty_like(input)
    dgamma = torch.empty_like(weight)
    dbeta = torch.empty_like(weight)
    return dX, dgamma, dbeta


@register_meta(aten.native_layer_norm.default)
def meta_ln(input: torch.Tensor, normalized_shape, weight, bias, eps):
    n_input = input.size(1)

    output = torch.empty_like(input)
    running_mean = torch.empty((n_input), device='meta')
    running_var = torch.empty((n_input), device='meta')
    return output, running_mean, running_var


@register_meta(aten.native_layer_norm_backward.default)
def meta_ln_backward(dY: torch.Tensor, input: torch.Tensor, normalized_shape, mean, rstd, weight, bias,
                     grad_input_mask):
    dX = torch.empty_like(input)
    dgamma = torch.empty_like(weight)
    dbeta = torch.empty_like(bias)
    return dX, dgamma, dbeta


@register_meta(aten._adaptive_avg_pool2d_backward.default)
def meta_adaptive_avg_pool2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
):
    grad_input = torch.empty_like(input)
    return torch.empty_like(input)


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
    import torch._refs as refs    # avoid import cycle in mypy

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

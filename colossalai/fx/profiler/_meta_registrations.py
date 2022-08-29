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
                name = (
                    op.__name__
                    if op._overloadname != "default"
                    else op.overloadpacket.__name__
                )
                meta_lib.impl(name, f)

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
                    )
                )
            else:
                ret_shape.append(
                    _formula(
                        dims[i], padding[i], dilation[i], kernel_size[i], stride[i]
                    )
                )
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
        shape_out = calc_conv_nd_return_shape(
            dims, kernel_size, stride, padding, dilation
        )
    out = input_tensor.new_empty((input_tensor.shape[0], out_channels, *shape_out))
    mem_fmt = pick_memory_format()
    out = out.to(memory_format=mem_fmt)  # type: ignore[call-overload]
    return out


@register_meta(aten.convolution_backward.default)
def meta_conv_backward(
    grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, 
    bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask
):
    return torch.empty_like(input), torch.empty_like(weight), torch.empty((bias_sizes), device='meta')


@register_meta(aten.relu.default)
def meta_relu(input: torch.Tensor):
    return torch.empty_like(input)


@register_meta(aten.native_batch_norm.default)
def meta_bn(
    input: torch.Tensor, 
    weight, bias, running_mean, running_var, training, momentum, eps
):
    n_input = input.size(1)

    output = torch.empty_like(input)
    running_mean = torch.empty((n_input), device='meta')
    running_var = torch.empty((n_input), device='meta')
    return output, running_mean, running_var

@register_meta(aten.native_layer_norm.default)
def meta_ln(
    input: torch.Tensor, 
    normalized_shape, weight, bias, eps
):
    n_input = input.size(1)

    output = torch.empty_like(input)
    running_mean = torch.empty((n_input), device='meta')
    running_var = torch.empty((n_input), device='meta')
    return output, running_mean, running_var


@register_meta(aten.native_layer_norm_backward.default)
def meta_ln_backward(
    dY: torch.Tensor,
    input: torch.Tensor, 
    normalized_shape, mean, rstd, weight, bias, grad_input_mask
):
    dX = torch.empty_like(input)
    dgamma = torch.empty_like(weight)
    dbeta = torch.empty_like(bias)
    return dX, dgamma, dbeta

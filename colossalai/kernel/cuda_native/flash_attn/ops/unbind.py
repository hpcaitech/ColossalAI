# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Sequence, Tuple, Union

import torch


def get_stack_strides(
    tensors: Sequence[torch.Tensor], dim: int
) -> Optional[Tuple[int, ...]]:
    """
    If the tensors are already stacked on dimension :code:`dim`, \
        returns the strides of the stacked tensors. \
        Otherwise returns :code:`None`.
    """
    if len(tensors) <= 1 or dim > tensors[0].ndim:
        return None

    final_stride = []
    for i in range(tensors[0].ndim + 1):
        if i == dim:
            final_stride.append(
                tensors[1].storage_offset() - tensors[0].storage_offset()
            )
            continue
        if i > dim:
            i -= 1
        final_stride.append(tensors[0].stride(i))

    storage_data_ptr: Optional[int] = None
    for i, x in enumerate(tensors[1:]):
        # Sanity checks
        if x.shape != tensors[0].shape:
            return None
        if x.stride() != tensors[0].stride():
            return None
        if (
            x.storage_offset()
            != tensors[0].storage_offset() + (i + 1) * final_stride[dim]
        ):
            return None
        if storage_data_ptr is None:
            storage_data_ptr = tensors[0].storage().data_ptr()
        # Actual storage check
        if x.storage().data_ptr() != storage_data_ptr:
            return None
    return tuple(final_stride)


def _stack_or_none_fw(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    dim: int,
) -> Optional[torch.Tensor]:
    strides = get_stack_strides(tensors, dim)
    if strides is not None:
        input_shape = list(tensors[0].shape)
        input_shape.insert(dim, len(tensors))
        return tensors[0].as_strided(input_shape, strides)
    return None


def _stack_fw(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    dim: int,
) -> torch.Tensor:
    out = _stack_or_none_fw(tensors, dim)
    if out is None:
        out = torch.stack(tensors, dim=dim)
    return out


class _Unbind(torch.autograd.Function):
    """
    See function `unbind`
    """

    @staticmethod
    # type: ignore
    def forward(ctx, x: torch.Tensor, dim: int):
        ctx.dim = dim
        return x.unbind(dim)

    @classmethod
    # type: ignore
    def backward(cls, ctx, *tensors: torch.Tensor):
        return _stack_fw(tensors, ctx.dim), None


class _StackOrNone(torch.autograd.Function):
    """
    See function `stack_or_none`
    """

    @staticmethod
    # type: ignore
    def forward(ctx, dim: int, *tensors: torch.Tensor):
        ctx.dim = dim
        return _stack_or_none_fw(tensors, dim=dim)

    @classmethod
    # type: ignore
    def backward(cls, ctx, grad: torch.Tensor):
        return (None, *grad.unbind(dim=ctx.dim))


def unbind(x: torch.Tensor, dim: int) -> Tuple[torch.Tensor, ...]:
    """
    Does exactly the same as :attr:`torch.unbind` for the forward.
    In backward, avoids a :attr:`torch.cat` if the gradients
    are already multiple views of the same storage
    """
    return _Unbind.apply(x, dim)


def stack_or_none(tensors: Sequence[torch.Tensor], dim: int) -> torch.Tensor:
    """
    Does exactly the same as :attr:`torch.stack` if the tensors can be concatenated
    without any memory operation. Otherwise returns None.
    """
    return _StackOrNone.apply(dim, *tensors)

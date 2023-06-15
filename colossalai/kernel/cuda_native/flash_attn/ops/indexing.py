# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional, Sequence

import torch

from .common import BaseOperator, get_xformers_operator, register_operator


@register_operator
class ScaledIndexAddFw(BaseOperator):
    OPERATOR = get_xformers_operator("scaled_index_addF")
    OPERATOR_CATEGORY = "indexing"
    NAME = "scaled_index_addF"


@register_operator
class ScaledIndexAddBw(BaseOperator):
    OPERATOR = get_xformers_operator("scaled_index_addB")
    OPERATOR_CATEGORY = "indexing"
    NAME = "scaled_index_addB"


@register_operator
class IndexSelect(BaseOperator):
    OPERATOR = get_xformers_operator("index_select")
    OPERATOR_CATEGORY = "indexing"
    NAME = "index_select"


class _ScaledIndexAdd(torch.autograd.Function):
    @staticmethod
    # type: ignore
    def forward(
        ctx,
        input: torch.Tensor,
        index: torch.Tensor,
        source: torch.Tensor,
        scaling: Optional[torch.Tensor],
        alpha: float,
    ) -> torch.Tensor:
        ScaledIndexAddFw.OPERATOR(
            output=input,  # in-place
            input=input,
            source=source,
            index=index,
            source_scaling=scaling,
            alpha=alpha,
        )
        ctx.mark_dirty(input)
        ctx.save_for_backward(index, scaling, source)
        ctx.source_shape = source.shape
        ctx.alpha = alpha
        return input

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        index, scaling, source = ctx.saved_tensors
        grad_source = torch.empty_like(grad_output[: index.shape[0]])
        grad_source_scaling = (
            torch.empty(
                ctx.source_shape,
                dtype=scaling.dtype,
                device=scaling.device,
            )
            if scaling is not None
            else None
        )
        ScaledIndexAddBw.OPERATOR(
            grad_source=grad_source,
            grad_source_scaling=grad_source_scaling,
            grad_output=grad_output,
            source=source,
            index=index,
            source_scaling=scaling,
            alpha=ctx.alpha,
        )
        if grad_source_scaling is not None:
            grad_source_scaling = grad_source_scaling.sum((0, 1))
        return (
            grad_output,  # input
            None,  # index
            grad_source,  # source
            grad_source_scaling,  # scaling
            None,  # alpha
        )


def scaled_index_add(
    input: torch.Tensor,  # [B, M, D]
    index: torch.Tensor,  # [Bi] - int64
    source: torch.Tensor,  # [Bi, M, D]
    scaling: Optional[torch.Tensor] = None,  # [D]
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    In-place scaling+index_add

    Indices in ``index`` are assumed to be unique

    :Note:

        The FW pass is done in-place (``input`` is modified)

    :Note:

        This is experimental and has only been optimized for a few shapes

    :Equivalent pytorch code:

    .. code-block:: python

        return torch.index_add(inp, dim=0, source=scaling * src, index=indices, alpha=alpha)
    """
    return _ScaledIndexAdd.apply(
        input,
        index,
        source,
        scaling,
        alpha,
    )


class _IndexSelectCat(torch.autograd.Function):
    @staticmethod
    # type: ignore
    def forward(
        ctx,
        *args: torch.Tensor,
    ) -> torch.Tensor:
        assert len(args) % 2 == 0
        sources = args[: len(args) // 2]
        indices = args[len(args) // 2 :]
        output_shape = 0
        total_source_elements = 0
        for source, index in zip(sources, indices):
            output_shape += index.shape[0] * source.shape[1]
            total_source_elements += source.shape[0] * source.shape[1]
        output = torch.empty(
            [output_shape], dtype=sources[0].dtype, device=sources[0].device
        )
        output_i = 0
        for source, index in zip(sources, indices):
            elements_here = index.shape[0] * source.shape[1]
            IndexSelect.OPERATOR(
                output=output[output_i : output_i + elements_here].view(
                    [index.shape[0], source.shape[1]]
                ),
                source=source,
                index=index,
            )
            output_i += elements_here
        ctx.save_for_backward(*indices)
        ctx.total_source_elements = total_source_elements
        ctx.source_shapes = [s.shape for s in sources]
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        indices = ctx.saved_tensors
        grad_sources = torch.zeros(
            [ctx.total_source_elements],
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        grad_sources_i = 0
        grad_output_i = 0
        gradients = []
        for source_shape, index in zip(ctx.source_shapes, indices):
            grad_output_slice = grad_output[
                grad_output_i : grad_output_i + index.shape[0] * source_shape[1]
            ].reshape([index.shape[0], source_shape[1]])
            grad_output_i += index.shape[0] * source_shape[1]

            gradient_source = grad_sources[
                grad_sources_i : grad_sources_i + source_shape[0] * source_shape[1]
            ].reshape(source_shape)
            grad_sources_i += source_shape[0] * source_shape[1]

            ScaledIndexAddFw.OPERATOR(
                output=gradient_source.unsqueeze(1),
                input=None,
                source=grad_output_slice.unsqueeze(1),
                index=index,
                source_scaling=None,
                alpha=1.0,
            )
            gradients.append(gradient_source)
        return (*gradients, *([None] * len(gradients)))


def index_select_cat(
    sources: Sequence[torch.Tensor], indices: Sequence[torch.Tensor]
) -> torch.Tensor:
    """
    Indices in ``index`` are assumed to be unique

    :Note:

        This is experimental and has only been optimized for a few shapes

    :Example:

    Given:
    - ``sources[0]`` of shape ``[S0, D0]``
    - ``indices[0]`` of shape ``[I0]``
    - ``sources[1]`` of shape ``[S1, D1]``
    - ``indices[1]`` of shape ``[I1]``
    returns a ``torch.Tensor`` of shape ``[I0 * D0 + I1 * D1]``

    :Equivalent pytorch code:

    .. code-block:: python

        return torch.cat([s[i.long()].flatten() for s, i in zip(sources, indices)], dim=0)
    """
    return _IndexSelectCat.apply(*sources, *indices)

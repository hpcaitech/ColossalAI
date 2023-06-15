# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import textwrap
from typing import List, Type, TypeVar

#from . import cutlass, flash, small_k, triton
from . import cutlass, flash, small_k
from .common import AttentionBwOpBase, AttentionFwOpBase, Inputs


def _is_cutlass_fwd_faster_than_flash(inp: Inputs) -> bool:
    # For dropout, we can't mix & match kernels
    # Unfortunately, the dropout implementation in CUTLASS
    # backward is pretty slow for the BW, so disable it here
    if inp.p > 0.0:
        return False

    # Large values of K
    return max(inp.query.shape[-1], inp.value.shape[-1]) > 64


def _is_triton_fwd_fastest(inp: Inputs) -> bool:
    # TODO: fill out
    return False


T = TypeVar("T", Type[AttentionFwOpBase], Type[AttentionBwOpBase])


def _format_inputs_description(inp: Inputs) -> str:
    return f"""query       : shape={tuple(inp.query.shape)} ({inp.query.dtype})
key         : shape={tuple(inp.key.shape)} ({inp.key.dtype})
value       : shape={tuple(inp.value.shape)} ({inp.value.dtype})
attn_bias   : {type(inp.attn_bias)}
p           : {inp.p}"""


def _ensure_op_supports_or_raise(exc_type, name: str, op, inp: Inputs) -> None:
    reasons = op.not_supported_reasons(inp)
    if not reasons:
        return
    raise exc_type(
        f"""Operator `{name}` does not support inputs:
{textwrap.indent(_format_inputs_description(inp), '     ')}
{_format_not_supported_reasons(op, reasons)}"""
    )


def _format_not_supported_reasons(op, reasons: List[str]) -> str:
    return f"`{op.NAME}` is not supported because:\n    " + "\n    ".join(reasons)


def _run_priority_list(name: str, priority_list: List[T], inp: Inputs) -> T:
    not_supported_reasons: List[List[str]] = []
    for op in priority_list:
        not_supported = op.not_supported_reasons(inp)
        if not not_supported:
            return op
        not_supported_reasons.append(not_supported)

    # Let's write a nice message explaining what we tried and why it's not supported
    msg = f"""No operator found for `{name}` with inputs:
{textwrap.indent(_format_inputs_description(inp), '     ')}"""
    for op, not_supported in zip(priority_list, not_supported_reasons):
        msg += "\n" + _format_not_supported_reasons(op, not_supported)
    raise NotImplementedError(msg)


def _dispatch_fw(inp: Inputs) -> Type[AttentionFwOpBase]:
    """Computes the best operator for forward

    Raises:
        NotImplementedError: if not operator was found

    Returns:
        AttentionOp: The best operator for the configuration
    """

    priority_list_ops: List[Type[AttentionFwOpBase]] = [
        flash.FwOp,
        #triton.FwOp,
        cutlass.FwOp,
        small_k.FwOp,
    ]
    if _is_cutlass_fwd_faster_than_flash(inp):
        priority_list_ops.remove(cutlass.FwOp)
        priority_list_ops.insert(0, cutlass.FwOp)
    # if _is_triton_fwd_fastest(inp):
    #     priority_list_ops.remove(triton.FwOp)
    #     priority_list_ops.insert(0, triton.FwOp)
    return _run_priority_list(
        "memory_efficient_attention_forward", priority_list_ops, inp
    )


def _is_cutlassB_faster_than_flash(inp: Inputs) -> bool:
    embed_dim = max(inp.query.shape[-1], inp.value.shape[-1])
    return embed_dim > 64 and inp.attn_bias is None and inp.p == 0.0


def _dispatch_bw(inp: Inputs) -> Type[AttentionBwOpBase]:
    priority_list_ops: List[Type[AttentionBwOpBase]] = [
        flash.BwOp,
        cutlass.BwOp,
        # CUDA illegal memory issues, race conditions etc..
        # triton.BwOp,
        # Deprecated
        small_k.BwOp,
    ]
    if _is_cutlassB_faster_than_flash(inp):
        priority_list_ops.remove(cutlass.BwOp)
        priority_list_ops.insert(0, cutlass.BwOp)
    return _run_priority_list(
        "memory_efficient_attention_backward", priority_list_ops, inp
    )

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, List, Mapping, Optional, Set, Tuple, Union

import torch

from ..common import get_xformers_operator, register_operator
from . import attn_bias
from .attn_bias import (
    AttentionBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
    LowerTriangularMask,
    LowerTriangularMaskWithTensorBias,
)
from .common import (
    AttentionBwOpBase,
    AttentionFwOpBase,
    Context,
    Gradients,
    Inputs,
    check_lastdim_alignment_stride1,
)


def _uses_tensorcores(sm: int, is_half: bool) -> bool:
    if sm >= 80:
        return True
    if sm >= 70:
        return is_half
    return False


def _minimum_gemm_alignment(inp: Inputs) -> int:
    if inp.device.type != "cuda":
        return 1
    cap = torch.cuda.get_device_capability(inp.device)
    sm = cap[0] * 10 + cap[1]
    bits_per_scalar = {torch.float: 32, torch.half: 16, torch.bfloat16: 16}[
        inp.query.dtype
    ]
    uses_tensorcores = _uses_tensorcores(sm, bits_per_scalar == 16)
    matmul_alignment_mn = 1
    if sm >= 80:
        matmul_alignment_mn = 4
    if uses_tensorcores:
        matmul_alignment_mn = max(matmul_alignment_mn, 128 // bits_per_scalar)
    return matmul_alignment_mn


def _get_seqlen_info(
    inp: Inputs,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int, int]:
    attn_bias = inp.attn_bias
    if isinstance(
        attn_bias, (BlockDiagonalMask, BlockDiagonalCausalWithOffsetPaddedKeysMask)
    ):
        if (
            isinstance(attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask)
            and attn_bias.causal_diagonal is not None
        ):
            attn_bias.causal_diagonal = attn_bias.causal_diagonal.to(inp.query.device)
        attn_bias.k_seqinfo.to(inp.query.device)
        attn_bias.q_seqinfo.to(inp.query.device)
        seqstart_k = attn_bias.k_seqinfo.seqstart
        seqstart_q = attn_bias.q_seqinfo.seqstart
        max_seqlen_q = attn_bias.q_seqinfo.max_seqlen
        max_seqlen_k = attn_bias.k_seqinfo.max_seqlen
    else:
        seqstart_k = None
        seqstart_q = None
        max_seqlen_q = -1
        max_seqlen_k = -1

    return seqstart_k, seqstart_q, max_seqlen_q, max_seqlen_k


def _get_tensor_bias(
    attn_bias: Optional[Union[torch.Tensor, AttentionBias]]
) -> Optional[torch.Tensor]:
    if isinstance(attn_bias, torch.Tensor):
        return attn_bias
    elif isinstance(attn_bias, LowerTriangularMaskWithTensorBias):
        return attn_bias._bias
    return None


def _check_bias_alignment(
    reasons: List[str], attn_bias: Optional[Union[torch.Tensor, AttentionBias]]
) -> None:
    attn_bias_tensor = _get_tensor_bias(attn_bias)
    if attn_bias_tensor is not None:
        alignment = 128 // torch.finfo(attn_bias_tensor.dtype).bits
        show_padding_hint = False
        for d in range(attn_bias_tensor.ndim - 1):
            if attn_bias_tensor.stride(d) % alignment != 0:
                reasons.append(
                    f"attn_bias.stride(-2) % {alignment} != 0 (attn_bias.stride() = {attn_bias_tensor.stride()})"
                )
                show_padding_hint = True
        if show_padding_hint:
            reasons.append(
                """\
HINT: To use an `attn_bias` with a sequence length that is not a multiple of 8, \
you need to ensure memory is aligned by slicing a bigger tensor. \
Example: use `attn_bias = torch.zeros([1, 1, 5, 8])[:,:,:,:5]` instead of `torch.zeros([1, 1, 5, 5])`"""
            )
        # We can have stride=0 sometimes if dimension=1
        if attn_bias_tensor.stride(-1) > 1:
            reasons.append(
                f"attn_bias.stride(-1) > 1 (attn_bias.stride() = {attn_bias_tensor.stride()}) - "
                "you should call `.contiguous()` on the bias"
            )


def _custom_mask_type(bias: Optional[Union[torch.Tensor, AttentionBias]]) -> int:
    if isinstance(
        bias,
        (
            LowerTriangularMask,
            BlockDiagonalCausalMask,
            BlockDiagonalCausalWithOffsetPaddedKeysMask,
        ),
    ):
        return 1
    if isinstance(bias, attn_bias.BlockDiagonalCausalFromBottomRightMask):
        return 2
    return 0


@register_operator
class FwOp(AttentionFwOpBase):
    """xFormers' MHA kernel based on CUTLASS.
    Supports a large number of settings (including without TensorCores, f32 ...)
    and GPUs as old as P100 (Sm60)
    """

    OPERATOR = get_xformers_operator("efficient_attention_forward_cutlass")
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.float, torch.half, torch.bfloat16}
    SUPPORTED_MAX_K = 65536
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {
        type(None),
        torch.Tensor,
        LowerTriangularMask,
        LowerTriangularMaskWithTensorBias,
        BlockDiagonalMask,
        BlockDiagonalCausalMask,
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
        attn_bias.BlockDiagonalCausalFromBottomRightMask,
    }
    SUPPORTS_DROPOUT = True
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = True
    NAME = "cutlassF"

    _TEST_K: List[int] = [
        32,  # 64x64 kernel
        128,  # 64x128 kernel
        256,  # 64x128 with accumulation in gmem
    ]

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        if type(inp.attn_bias) not in FwOp.SUPPORTED_ATTN_BIAS_TYPES:
            raise NotImplementedError("Unsupported attn_bias type")
        seqstart_k, seqstart_q, max_seqlen_q, _ = _get_seqlen_info(inp)
        out, lse, rng_seed, rng_offset = cls.OPERATOR(
            query=inp.query,
            key=inp.key,
            value=inp.value,
            attn_bias=_get_tensor_bias(inp.attn_bias),
            seqstart_q=seqstart_q,
            seqstart_k=seqstart_k,
            max_seqlen_q=max_seqlen_q,
            dropout_p=inp.p,
            compute_logsumexp=needs_gradient,
            custom_mask_type=_custom_mask_type(inp.attn_bias),
            scale=inp.scale,
            causal_diagonal=inp.attn_bias.causal_diagonal
            if isinstance(inp.attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask)
            else None,
            seqlen_k=inp.attn_bias.k_seqinfo.seqlen
            if isinstance(inp.attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask)
            else None,
        )
        ctx: Optional[Context] = None
        if needs_gradient:
            ctx = Context(
                out=out,
                lse=lse,
                # cutlass forward is only compatible with cutlass backward if
                # dropout is used (because of the way RNG states are passed and the
                # way random numbers are generated during backward)
                op_bw=BwOp if inp.p != 0 else None,
            )
            if inp.p != 0:
                ctx.rng_state = torch.tensor(
                    [rng_seed, rng_offset], dtype=torch.int64, device="cpu"
                )
        return out, ctx

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(FwOp, cls).not_supported_reasons(d)
        matmul_alignment_mn = _minimum_gemm_alignment(d)
        check_lastdim_alignment_stride1(reasons, "query", d.query, matmul_alignment_mn)
        check_lastdim_alignment_stride1(reasons, "value", d.value, matmul_alignment_mn)
        _check_bias_alignment(reasons, d.attn_bias)
        return reasons

    @classmethod
    # type: ignore
    def operator_flop(
        cls,
        q,
        k,
        v,
        b,
        seqstart_q,
        seqstart_k,
        max_seqlen_q_,
        compute_lse,
        custom_mask_type,
        *a,
    ) -> int:
        return cls.attn_operator_flop(
            q,
            k,
            v,
            causal=custom_mask_type > 0,
            seqstart_k=seqstart_k,
            seqstart_q=seqstart_q,
        )


@register_operator
class BwOp(AttentionBwOpBase):
    __doc__ = FwOp.__doc__

    OPERATOR = get_xformers_operator("efficient_attention_backward_cutlass")
    SUPPORTED_DEVICES = FwOp.SUPPORTED_DEVICES
    SUPPORTED_DTYPES = FwOp.SUPPORTED_DTYPES
    SUPPORTED_MAX_K = FwOp.SUPPORTED_MAX_K
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {
        type(None),
        torch.Tensor,
        LowerTriangularMask,
        # TODO: Fix handling of gradient through the fMHA autograd function
        # LowerTriangularMaskWithTensorBias,
        BlockDiagonalMask,
        BlockDiagonalCausalMask,
        attn_bias.BlockDiagonalCausalFromBottomRightMask,
    }
    SUPPORTS_ATTN_BIAS_GRAD = True
    SUPPORTS_DROPOUT = FwOp.SUPPORTS_DROPOUT
    SUPPORTS_CUSTOM_SCALE = FwOp.SUPPORTS_CUSTOM_SCALE
    SUPPORTS_DIFFERENT_VALUE_EMBED = FwOp.SUPPORTS_DIFFERENT_VALUE_EMBED
    NAME = "cutlassB"

    ERROR_ATOL: Mapping[torch.dtype, float] = {
        torch.float: 5e-4,
        # increased from 9e-2, more opportunities for numerical errors when bias is
        # used, noticed in gK on SM80
        torch.half: 1e-1,
        torch.bfloat16: 7e-1,
    }

    _TEST_K: List[int] = [
        32,  # 64x64 kernel
        128,  # 64x128/128x128 kernel
        256,  # 64x128 with accumulation in gmem
    ]

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(BwOp, cls).not_supported_reasons(d)
        matmul_alignment_mn = _minimum_gemm_alignment(d)

        check_lastdim_alignment_stride1(reasons, "query", d.query, matmul_alignment_mn)
        check_lastdim_alignment_stride1(reasons, "key", d.key, matmul_alignment_mn)
        check_lastdim_alignment_stride1(reasons, "value", d.value, matmul_alignment_mn)
        _check_bias_alignment(reasons, d.attn_bias)
        attn_bias_tensor = _get_tensor_bias(d.attn_bias)

        # Backprop of gradient through broadcasted bias is not supported
        if attn_bias_tensor is not None and attn_bias_tensor.requires_grad:
            # Don't forget that inputs are either in BMK or BMHK!
            if d.query.ndim == 3 and attn_bias_tensor.ndim == 3:
                expected_bias_shape = (*d.query.shape[:2], d.key.shape[1])
            else:
                # bias is B H Mq Mk
                expected_bias_shape = (
                    d.query.shape[0],
                    d.query.shape[2] if d.query.ndim == 4 else 1,
                    d.query.shape[1],
                    d.key.shape[1],
                )
            if tuple(attn_bias_tensor.shape) != expected_bias_shape:
                reasons.append(
                    "Broadcasting the `attn_bias` tensor is not supported "
                    f"(shape: {tuple(attn_bias_tensor.shape)}"
                    f"/ expected: {expected_bias_shape})"
                )
        return reasons

    @classmethod
    def apply(cls, ctx: Context, inp: Inputs, grad: torch.Tensor) -> Gradients:
        if type(inp.attn_bias) not in BwOp.SUPPORTED_ATTN_BIAS_TYPES:
            raise NotImplementedError("Unsupported attn_bias type")

        seqstart_k, seqstart_q, max_seqlen_q, max_seqlen_k = _get_seqlen_info(inp)
        dtype = inp.query.dtype

        rng_seed = rng_offset = 0
        if inp.p != 0.0:
            if (
                ctx.rng_state is None
                or ctx.rng_state.dtype != torch.int64
                or ctx.rng_state.device.type != "cpu"
                or ctx.rng_state.shape != (2,)
            ):
                raise NotImplementedError(f"Invalid rng_state: {ctx.rng_state}")
            rng_seed, rng_offset = ctx.rng_state.tolist()

        force_pad_inf = torch.cuda.get_device_capability(inp.query.device) == (7, 5)
        (grad_q, grad_k, grad_v, grad_bias) = cls.OPERATOR(
            grad.to(dtype),
            inp.query,
            inp.key,
            inp.value,
            _get_tensor_bias(inp.attn_bias),
            cu_seqlens_q=seqstart_q,
            cu_seqlens_k=seqstart_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            logsumexp=ctx.get_padded_lse(32, force_pad_inf=force_pad_inf),
            output=ctx.out.to(dtype),
            dropout_p=inp.p,
            # if not using dropout, seed and offset are irrelevant but still expected
            # in function signature so just pass 0
            # seed and offset could be None if a different FW op other than cutlass
            # was used.
            rng_seed=rng_seed,
            rng_offset=rng_offset,
            custom_mask_type=_custom_mask_type(inp.attn_bias),
            scale=inp.scale,
            num_splits_key=-1,  # Let C++ determine it
        )

        # c++/CUDA implementation returns an uninitialized tensor if bias doesn't
        # require grad
        if not (
            isinstance(inp.attn_bias, torch.Tensor) and inp.attn_bias.requires_grad
        ):
            grad_bias = None

        return Gradients(dq=grad_q, dk=grad_k, dv=grad_v, db=grad_bias)

    @classmethod
    # type: ignore
    def operator_flop(
        cls,
        dO,
        q,
        k,
        v,
        b,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        logsumexp,
        output,
        dropout_p,
        rng_seed,
        rng_offset,
        custom_mask_type,
        scale,
    ) -> int:
        return cls.attn_operator_flop(
            q,
            k,
            v,
            seqstart_q=cu_seqlens_q,
            seqstart_k=cu_seqlens_k,
            causal=custom_mask_type > 0,
        )

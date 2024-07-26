from enum import Enum
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange

from colossalai.kernel.kernel_loader import (
    FlashAttentionForFloatAndCustomMaskLoader,
    FlashAttentionLoader,
    FlashAttentionWithCustomMaskLoader,
    KernelLoader,
)

from .utils import RingComm, get_half_index, split_varlen_zigzag

__all__ = [
    "AttnMaskType",
    "ColoAttention",
]

_flash_attn_forward = _flash_attn_backward = None
_unpad_input = _pad_input = None


class AttnMaskType(Enum):
    CUSTOM = 0
    PADDED = 1
    CAUSAL = 2
    PADDED_CAUSAL = 3


def invert_mask(mask: torch.Tensor) -> torch.Tensor:
    """Invert the mask tensor.

    Args:
        mask (torch.Tensor): Mask tensor. Shape should be [B, 1, Sq, Sq]

    Returns:
        torch.Tensor: Inverted mask tensor.
    """
    inverted_mask = 1.0 - mask
    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(mask.dtype).min)


# adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py
def get_pad_info(
    padding_mask: torch.Tensor, invert: Optional[bool] = False, return_indices: Optional[bool] = True
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """Get padding information from padding mask.

    Args:
        padding_mask (torch.Tensor): Padding mask tensor. Shape should be [B, Sq]
        invert (Optional[bool], optional): Whether to reverse the padding mask.
        return_indices (Optional[bool], optional): Whether to return the indices of non-masked tokens.

    Returns:
        max_seqlen_in_batch (int): Maximum sequence length in the batch.
        cu_seqlens (torch.Tensor): Shape [B+1]. Cumulative sequence lengths of the sequences in the batch.
        indices (torch.Tensor): Shape [total_nonzero]. The indices of non-masked tokens from the flattened input sequence.
    """
    if invert:
        padding_mask = padding_mask.logical_not()
    seqlens_in_batch = padding_mask.sum(dim=-1, dtype=torch.int32)
    if return_indices:
        indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()

    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    if return_indices:
        return max_seqlen_in_batch, cu_seqlens, indices
    return max_seqlen_in_batch, cu_seqlens


class ColoAttention:
    _kernel_dispatch_map: Optional[Dict[torch.dtype, Dict[Optional[AttnMaskType], Callable]]] = None

    @staticmethod
    def _init_kernels_dispatch():
        if ColoAttention._kernel_dispatch_map is None:
            # fp16/bf16
            half_dispatch_map = {
                None: FlashAttentionLoader(),
                AttnMaskType.CUSTOM: FlashAttentionWithCustomMaskLoader(),
                AttnMaskType.PADDED: FlashAttentionLoader(),
                AttnMaskType.CAUSAL: FlashAttentionLoader(),
                AttnMaskType.PADDED_CAUSAL: FlashAttentionLoader(),
            }
            # fp32
            float_dispatch_map = {
                None: FlashAttentionForFloatAndCustomMaskLoader(),
                AttnMaskType.CUSTOM: FlashAttentionForFloatAndCustomMaskLoader(),
                AttnMaskType.PADDED: FlashAttentionForFloatAndCustomMaskLoader(),
                AttnMaskType.CAUSAL: FlashAttentionForFloatAndCustomMaskLoader(),
                AttnMaskType.PADDED_CAUSAL: FlashAttentionForFloatAndCustomMaskLoader(),
            }
            ColoAttention._kernel_dispatch_map = {
                torch.float16: half_dispatch_map,
                torch.bfloat16: half_dispatch_map,
                torch.float32: float_dispatch_map,
            }

    @staticmethod
    def _dispatch_kernel(dtype: torch.dtype, mask_type: Optional[AttnMaskType]) -> Callable:
        ColoAttention._init_kernels_dispatch()
        if (
            dtype not in ColoAttention._kernel_dispatch_map
            or mask_type not in ColoAttention._kernel_dispatch_map[dtype]
        ):
            raise ValueError(
                "FlashAttention kernel is not available for dtype {} and mask_type {}".format(dtype, mask_type)
            )
        # lazy load
        if isinstance(ColoAttention._kernel_dispatch_map[dtype][mask_type], KernelLoader):
            ColoAttention._kernel_dispatch_map[dtype][mask_type] = ColoAttention._kernel_dispatch_map[dtype][
                mask_type
            ].load()
        return ColoAttention._kernel_dispatch_map[dtype][mask_type]

    @staticmethod
    def prepare_attn_kwargs(
        shape_4d: Tuple[int],
        dtype: torch.dtype,
        device: torch.device,
        q_padding_mask: Optional[torch.Tensor] = None,
        kv_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Return a dictionary of keyword arguments for attention function. It supports 4 mask type.
        1. custom mask: no padding mask and is_causal=False, return {}, users should handle attention mask by themselves.
        2. padded mask: recv padding mask and is_causal=False, return {attention_mask, attention_mask_type, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, q_indices, kv_indices}.
        3. causal mask: no padding mask and is_causal=True, return {attention_mask, attention_mask_type}.
        4. padded causal mask: recv padding mask and is_causal=True, return {attention_mask, attention_mask_type, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, q_indices, kv_indices}.

        Args:
            shape_4d (Tuple[int]): Should be (B, 1, Sq, Skv)
            dtype (torch.dtype): Dtype of attention mask, generally should be ``hidden_states.dtype``
            device (torch.device): Device of attention mask, generally should be ``hidden_states.device``
            q_padding_mask (Optional[torch.Tensor], optional): Padding mask of query. It should be a long tensor or int tensor.
                The shape should be [B, Sq]. ``1`` means valid token, and ``0`` means padding token. Defaults to None.
            kv_padding_mask (Optional[torch.Tensor], optional): Padding mask of key and value. It should be a long tensor or int tensor.
                The shape should be [B, Skv]. ``1`` means valid token, and ``0`` means padding token.
                If it's None and ``q_padding_mask`` is not None, it will be set to ``q_padding_mask``. Defaults to None.
            is_causal (bool, optional): Whether to use causal attention mask. Defaults to False.
        Returns:
            Dict[str, torch.Tensor]: Dictionary of keyword arguments for attention function.
        """
        if q_padding_mask is None and not is_causal:
            return {}
        assert len(shape_4d) == 4 and shape_4d[1] == 1
        b, _, s_q, s_kv = shape_4d
        outputs = {}
        if (q_padding_mask is None or q_padding_mask.bool().all()) and (
            kv_padding_mask is None or kv_padding_mask.bool().all()
        ):
            # no padding
            assert is_causal
            outputs["attention_mask_type"] = AttnMaskType.CAUSAL
            attention_mask = torch.ones(s_q, s_kv, dtype=dtype, device=device)
            if s_q != 1:
                attention_mask = attention_mask.tril(diagonal=0)
            attention_mask = attention_mask.expand(b, s_q, s_kv)
        else:
            max_seqlen_q, cu_seqlens_q, q_indices = get_pad_info(q_padding_mask)
            if kv_padding_mask is None:
                # self attention
                kv_padding_mask = q_padding_mask
                max_seqlen_kv, cu_seqlens_kv, kv_indices = max_seqlen_q, cu_seqlens_q, q_indices
            else:
                max_seqlen_kv, cu_seqlens_kv, kv_indices = get_pad_info(kv_padding_mask)
            assert kv_padding_mask.shape == (
                b,
                s_kv,
            ), f"q_padding_mask shape {kv_padding_mask.shape} should be the same. ({shape_4d})"
            attention_mask = kv_padding_mask[:, None, :].expand(b, s_q, s_kv).to(dtype=dtype, device=device)
            outputs.update(
                {
                    "cu_seqlens_q": cu_seqlens_q,
                    "cu_seqlens_kv": cu_seqlens_kv,
                    "max_seqlen_q": max_seqlen_q,
                    "max_seqlen_kv": max_seqlen_kv,
                    "q_indices": q_indices,
                    "kv_indices": kv_indices,
                }
            )
            if is_causal:
                outputs["attention_mask_type"] = AttnMaskType.PADDED_CAUSAL
                if s_q != 1:
                    attention_mask = attention_mask * attention_mask.new_ones(s_q, s_kv).tril(diagonal=0)
            else:
                outputs["attention_mask_type"] = AttnMaskType.PADDED
        attention_mask = invert_mask(attention_mask).unsqueeze(1)
        outputs["attention_mask"] = attention_mask
        return outputs

    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_type: AttnMaskType = AttnMaskType.CUSTOM,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        q_indices: Optional[torch.Tensor] = None,
        kv_indices: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        scale: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Flash Attention function. It supports 4 mask type.
        1. custom mask: recv attention_mask
        2. padded mask: recv attention_mask, attention_mask_type, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, indices
        3. causal mask: recv attention_mask, attention_mask_type
        4. padded causal mask: recv attention_mask, attention_mask_type, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, indices

        Args:
            q (torch.Tensor): Query tensor. Shape should be [B, nHeads, Sq, D]
            k (torch.Tensor): Key tensor. Shape should be [B, nHeads, Sq, D]
            v (torch.Tensor): Value tensor. Shape should be [B, nHeads, Sq, D]
            attention_mask (Optional[torch.Tensor], optional): Attention mask tensor. Shape should be [B, 1, Sq, Sq]. Defaults to None.
            attention_mask_type (AttnMaskType, optional): Attention mask type. Defaults to AttnMaskType.CUSTOM.
            cu_seqlens_q (Optional[torch.Tensor], optional): The cumulative sequence lengths
                of the sequences in the batch, used to index into q.
                Shape should be [B+1]. Defaults to None.
            cu_seqlens_kv (Optional[torch.Tensor], optional): The cumulative sequence lengths
                of the sequences in the batch, used to index into kv.
                Shape should be [B+1]. Defaults to None.
            max_seqlen_q (Optional[int], optional): Maximum query sequence length in the batch. Defaults to None.
            max_seqlen_kv (Optional[int], optional): Maximum key/value sequence length in the batch. Defaults to None.
            indices (Optional[torch.Tensor], optional): The indices of non-masked tokens from the flattened input sequence.
                Shape should be [NUM_TOKENS]. Defaults to None.
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.
            scale (Optional[float], optional): Scaling factor applied prior to softmax. Defaults to None.

        Returns:
            torch.Tensor: Output tensor. Shape should be [B, nHeads, Sq, D]
        """
        # known issue: sdpa does not support attention mask which contains whole row of masked tokens, which leads to nan
        # this case is usaul when padding mask is used and self attention is performed
        # thus, we don't use sdpa when padding mask is used
        # sanity check
        if attention_mask is not None:
            assert torch.is_floating_point(attention_mask), "attention_mask should be a floating point tensor."
            if attention_mask_type in (AttnMaskType.CUSTOM, AttnMaskType.CAUSAL):
                assert (
                    cu_seqlens_q is None
                    and cu_seqlens_kv is None
                    and max_seqlen_q is None
                    and max_seqlen_kv is None
                    and q_indices is None
                    and kv_indices is None
                )
                if attention_mask_type == AttnMaskType.CUSTOM:
                    assert not torch.all(attention_mask != 0, dim=-1).any()
            elif attention_mask_type in (
                AttnMaskType.PADDED,
                AttnMaskType.PADDED_CAUSAL,
            ):
                assert (
                    cu_seqlens_q is not None
                    and cu_seqlens_kv is not None
                    and max_seqlen_q is not None
                    and max_seqlen_kv is not None
                    and q_indices is not None
                    and kv_indices is not None
                )
        # kernel dispatch
        mask_type = attention_mask_type if attention_mask is not None else None
        attn_func = ColoAttention._dispatch_kernel(q.dtype, mask_type)
        is_causal = attention_mask is not None and attention_mask_type in (
            AttnMaskType.CAUSAL,
            AttnMaskType.PADDED_CAUSAL,
        )
        return attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            scale=scale,
            attention_mask=attention_mask,
            is_causal=is_causal,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            q_indices=q_indices,
            kv_indices=kv_indices,
        )


def _load_varlen_helpers():
    """Helper to load functions for padding and unpadding packed sequences.
    Use only when flash attn is installed
    """
    global _pad_input, _unpad_input
    # Flash attn claims this is more efficient than torch's bool indexing due to avoiding
    # broadcast
    if _pad_input is None or _unpad_input is None:
        try:
            from flash_attn.bert_padding import index_first_axis, pad_input

            def unpad_input(hidden_states: torch.Tensor, indices: torch.Tensor):
                return index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices)

            _pad_input = pad_input
            _unpad_input = unpad_input
        except ImportError as e:
            raise RuntimeError(
                f"Flash Attention is not installed. You can install it via 'pip install flash-attn --no-build-isolation'"
            ) from e


def _load_flash_attn():
    """A light-weight loader to check whether flash-attn is installed.
    Can't use ColoAttention._dispatch_kernel because we mutate the backward pass
    """
    global _flash_attn_forward, _flash_attn_backward
    if _flash_attn_forward is None or _flash_attn_backward is None:
        try:
            from flash_attn.flash_attn_interface import _flash_attn_varlen_backward as _flash_attn_backward
            from flash_attn.flash_attn_interface import _flash_attn_varlen_forward as _flash_attn_forward
        except ImportError as e:
            raise RuntimeError(
                f"Flash Attention is not installed. You can install it via 'pip install flash-attn --no-build-isolation'"
            ) from e

    _load_varlen_helpers()


@triton.jit
def _rescale_out_lse_kernel(
    out_ptr,
    out_per_step_ptr,
    lse_ptr,
    lse_step_ptr,
    D,  # Each thread handles D elements
    stride_out_0,
    stride_out_1,
    stride_out_2,
    stride_out_per_step_0,
    stride_out_per_step_1,
    stride_out_per_step_2,
    stride_lse_0,
    stride_lse_1,
    BLOCK_M: tl.constexpr,
):
    batch_id = tl.program_id(0)
    sq_id = tl.program_id(1)
    h_id = tl.program_id(2)
    d_id = tl.arange(0, D)

    out_idx = batch_id * stride_out_0 + sq_id * stride_out_1 + h_id * stride_out_2 + d_id
    out_per_step_idx = batch_id * stride_out_per_step_0 + sq_id * stride_out_per_step_1 + h_id * stride_out_per_step_2
    lse_idx = batch_id * stride_lse_0 + h_id * stride_lse_1
    lse_step_idx = batch_id * stride_lse_0 + h_id * stride_lse_1

    # Load inputs
    out = tl.load(out_ptr + out_idx)
    out_per_step = tl.load(out_per_step_ptr + out_per_step_idx)
    lse = tl.load(lse_ptr + lse_idx)
    lse_step = tl.load(lse_step_ptr + lse_step_idx)

    # Element-wise rescale
    new_lse = lse + tl.log(1 + tl.exp(lse_step - lse))
    out = tl.exp(lse - new_lse) * out + tl.exp(lse_step - new_lse) * out_per_step

    tl.store(out_ptr + out_idx, out)
    tl.store(lse_ptr + lse_idx, new_lse)


def _rescale_out_lse_triton(out, block_out, lse, block_lse):
    T, H, D = out.shape

    assert out.is_contiguous() and block_out.is_contiguous() and lse.is_contiguous() and block_lse.is_contiguous()

    grid = lambda META: (triton.cdiv(T, META["BLOCK_M"]), H)
    _rescale_out_lse_kernel[grid](
        out,
        block_out,
        lse,
        block_lse,
        T,
        H,
        D,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        block_out.stride(0),
        block_out.stride(1),
        block_out.stride(2),
        lse.stride(0),
        lse.stride(1),
    )


def _rescale_out_lse(out, block_out, lse, block_lse):
    """
    Compute the new attention denominator:
        exp(lse) + exp(block_lse) = exp(max_scale) * (exp(min_scale - max_scale) + 1)
    Args:
        out: (B, Sq, H, D)
        block_out: (B, Sq, H, D)
        lse: (B, H, Sq, 1)
        block_lse: (B, H, Sq, 1)
    """

    # min_scale = torch.min(lse, block_lse)
    # max_scale = torch.max(lse, block_lse)
    # lse.data = max_scale + torch.log(1 + torch.exp(min_scale - max_scale))

    new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))

    new_block_lse = torch.exp(block_lse - lse)
    out.copy_(torch.exp(lse - new_lse) * out + new_block_lse * block_out)
    lse.copy_(new_lse)

    # See https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    # out.data = (out - F.sigmoid(block_lse - lse) * (out - block_out))
    # lse.data = (lse - F.logsigmoid(lse - block_lse))

    assert not (lse.isnan().any() or lse.isinf().any()), f"lse is nan: {lse}"


class RingAttention(torch.autograd.Function):
    """Implements the Ring Attention from `Ring Attention with Blockwise Transformers for Near-Infinite Context`
    (https://arxiv.org/abs/2310.01889).
    For load-balancing we adopted the "zigzag" attention scheme from https://github.com/zhuzilin/ring-flash-attention/tree/main
    For portable integration with more models, we don't follow the spirit of "block-wise FNN" in the original paper,
    which requires fusing FFN with the Flash Attention kernel/function (see https://arxiv.org/pdf/2305.19370;
    implemented in Jax and not optimized).
    """

    # Globle cache to avoid recomputation for same-lengthed sequences
    CU_SEQLENS: torch.Tensor = None  # [B+1]
    TOTAL_SEQLEN: int = None
    SUPPORTED_MASK_TYPES = (AttnMaskType.CAUSAL, AttnMaskType.PADDED_CAUSAL)

    @staticmethod
    def attention(
        q,  # (B, H, Sq, D)
        k,
        v,
        sp_group,
        sp_stream,
        attention_mask_type,
        cu_seqlens=None,
        max_seqlen=None,
        valid_indices=None,
        dropout_p=0,
        softmax_scale=None,
        deterministic=False,
        return_softmax=False,
        pad_output=True,
        **kwargs,
    ):
        """
        Ring Attention forward pass supporting variable-length sequences. When using varlen mode,
        each sequence in the batch should have length divisible by sp_size * 2.

        Args:
            q (torch.Tensor): Query tensor. Shape should be [B, nHeads, Sq, D]
            k (torch.Tensor): Key tensor. Shape should be [B, nHeads, Sq, Sq, D]
            v (torch.Tensor): Value tensor. Shape should be [B, nHeads, Sq, Sq, D]
            sp_group (Optional[dist.ProcessGroup]): Process group for sequence parallelism
            sp_tream (torch.cuda.Stream): An different stream for output correction.
            cu_seqlens (Optional[torch.Tensor], optional): The cumulative sequence lengths
                of the sequences in the batch, used to index into q.
                Shape should be [B+1].
            max_seqlen (Optional[int], optional): Maximum query sequence length in the batch.
            valid_indices (Optional[torch.Tensor], optional): The indices of non-masked tokens from get_pad_info.
                Shape should be [t].
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.
            softmax_scale (Optional[float], optional): Scaling factor applied prior to softmax.
            deterministic (bool, optional): Whether to force deterministic backward pass. See https://github.com/Dao-AILab/flash-attention/issues/349
            return_softmax (bool, optional): Whether to return the softmax denominator (logsumexp).
            pad_output: (bool, optional): If True, return a batch of sequences in the casual padded mode. Else,
                return a packed sequence.

        Returns:
            out: Output tensor of shape [B, nHeads, Sq, D] or [T, nHeads, D] if pad_output is False.
            softmax_lse: (if return_softmax is True) Softmax denominator (logsumexp).
                Shape should be [total_q_seqlen, nHeads]
        """
        _load_flash_attn()
        assert (
            q.shape[2] == k.shape[2]
        ), "Q, K and V having different sequence lengths (inference or cross-attn)\
            is not supported yet in training."
        assert (
            attention_mask_type in RingAttention.SUPPORTED_MASK_TYPES
        ), f"Mask type {attention_mask_type} is not supported yet."

        # (B, H, Sq, D) -> (B, Sq, H, D)
        q, k, v = [x.transpose(1, 2).contiguous() for x in (q, k, v)]
        b, sq, h, d = q.shape

        # Get sequence length info for varlen forward
        if attention_mask_type == AttnMaskType.CAUSAL:
            # All sequences share the same length
            # Cache to avoid recreation for a single sequence
            if b * sq == RingAttention.TOTAL_SEQLEN:
                cu_seqlens = RingAttention.CU_SEQLENS
                max_seqlen = RingAttention.TOTAL_SEQLEN
            else:
                RingAttention.CU_SEQLENS = cu_seqlens = torch.arange(
                    0, b * sq + 1, sq, device=q.device, dtype=torch.int32
                )
                RingAttention.TOTAL_SEQLEN = max_seqlen = b * sq

        # "Packed" mode where sequences of different lengths are packed into [total_q_seqlen, H, D]
        elif attention_mask_type == AttnMaskType.PADDED_CAUSAL:
            assert (
                cu_seqlens is not None and max_seqlen is not None and valid_indices is not None
            ), "Packed mode requires pre-computed cu_seqlens and max_seq_len."
            q, k, v = [_unpad_input(x, valid_indices) for x in (q, k, v)]

        out, softmax_lse = RingAttention.apply(
            q,
            k,
            v,
            sp_group,
            sp_stream,
            cu_seqlens,
            max_seqlen,
            dropout_p,
            softmax_scale,
            deterministic,
            return_softmax,
            attention_mask_type == AttnMaskType.PADDED_CAUSAL,
        )

        if attention_mask_type == AttnMaskType.PADDED_CAUSAL:
            if pad_output:
                out = _pad_input(out, valid_indices, b, sq)  # (T, ...) -> (B, Sq, ...)
                out = out.transpose(1, 2)  # (B, Sq, H, D) -> (B, H, Sq, D)
        else:
            out = out.transpose(1, 2)

        if return_softmax:
            return out, softmax_lse
        return out

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sp_group: dist.ProcessGroup,
        sp_stream: torch.cuda.Stream,
        cu_seqlens: Optional[torch.Tensor],
        max_seqlen: Optional[int],
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        deterministic: bool = False,
        return_softmax: bool = False,
        is_packed: bool = False,
    ):
        cu_seqlens_q = cu_seqlens_kv = cu_seqlens
        max_seqlen_q = max_seqlen_kv = max_seqlen
        misc_kwargs = {
            "window_size": (-1, -1),
            "alibi_slopes": None,
            "softmax_scale": q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale,
            "dropout_p": dropout_p,
            "block_table": None,
            "softcap": 0.0,
            "return_softmax": False,
        }
        if is_packed:
            t, h, d = q.shape
            # half of each seq
            half_idx_front = get_half_index(cu_seqlens, front=True)
            half_idx_back = get_half_index(cu_seqlens, front=False)
        else:
            b, sq, h, d = q.shape
            t = b * sq
            q, k, v = [x.view(t, h, d) for x in (q, k, v)]
        half_cu_seqlens = cu_seqlens // 2
        half_max_seqlen = max_seqlen // 2

        kv_comms = [RingComm(sp_group) for _ in range(2)]
        sp_size = kv_comms[0].world_size
        sp_rank = kv_comms[0].rank

        # Pre-allocate double buffer for overlapping and receiving next step's inputs
        kv_buffers = [torch.stack((k, v))]  # (2, B, Sq, H, D)
        kv_buffers.append(None)

        # outputs
        out = None
        block_out = [None, None]
        softmax_lse = [None, None]
        block_softmax_lse = [None, None]  # log sum exp, the denominator of softmax in attention
        rng_states = [None for _ in range(sp_size)]
        sp_streams = [torch.cuda.current_stream(), sp_stream]
        correction_done = torch.cuda.Event()

        # Overlap output correction with next flash attn
        for i in range(sp_size):
            with torch.cuda.stream(sp_streams[i % 2]):
                if i < sp_size - 1:
                    # Avoid overwriting the kv block used by the last flash attn call
                    kv_buffers[(i + 1) % 2] = torch.empty_like(kv_buffers[i % 2])
                # Wait for current kv from prev rank
                # NOTE: waiting outside the current stream will NOT correctly synchronize.
                kv_comms[(i + 1) % 2].wait()
                if i < sp_size - 1:
                    kv_comms[i % 2].send_recv(kv_buffers[i % 2], kv_buffers[(i + 1) % 2])

                if i == 0:
                    # Compute with local KV; no mask
                    q_block = q
                    kv_block = kv_buffers[i % 2]
                    (
                        _,
                        _,
                        _,
                        _,
                        block_out[i % 2],  # (B, Sq, H, D)
                        block_softmax_lse[i % 2],  # (H, total_q_seqlen)
                        _,
                        rng_states[i],
                    ) = _flash_attn_forward(
                        q_block,
                        kv_block[0],
                        kv_block[1],
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        causal=True,
                        **misc_kwargs,
                    )
                elif i <= sp_rank:
                    # Received the "surrounding" kv chunks
                    # Drop the second half of received kv
                    q_block = q
                    # (2, t // 2, H, D)
                    kv_block = kv_buffers[i % 2]
                    if is_packed:
                        kv_block = kv_block[:, half_idx_front]
                    else:
                        kv_block = kv_block[:, : t // 2]

                    (
                        _,
                        _,
                        _,
                        _,
                        block_out[i % 2],  # (B, Sq, H, D)
                        block_softmax_lse[i % 2],  # (H, total_q_seqlen)
                        _,
                        rng_states[i],
                    ) = _flash_attn_forward(
                        q_block,
                        kv_block[0],
                        kv_block[1],
                        cu_seqlens_q,
                        half_cu_seqlens,
                        max_seqlen_q,
                        half_max_seqlen,
                        causal=False,
                        **misc_kwargs,
                    )

                else:
                    # Received the inner kv chunks
                    # Drop the first half of q
                    if is_packed:
                        q_block = q[half_idx_back]
                    else:
                        q_block = q[t // 2 :]

                    kv_block = kv_buffers[i % 2]
                    (
                        _,
                        _,
                        _,
                        _,
                        block_out[i % 2],  # (B, Sq // 2, H, D)
                        block_softmax_lse[i % 2],  # (H, total_q_seqlen)
                        _,
                        rng_states[i],
                    ) = _flash_attn_forward(
                        q_block,
                        kv_block[0],
                        kv_block[1],
                        half_cu_seqlens,
                        cu_seqlens_kv,
                        half_max_seqlen,
                        max_seqlen_kv,
                        causal=False,
                        **misc_kwargs,
                    )

                block_out[i % 2] = block_out[i % 2].view(-1, h, d)
                block_softmax_lse[i % 2] = (
                    block_softmax_lse[i % 2].transpose(0, 1).unsqueeze(-1).contiguous().float()
                )  # (H, T) -> (T, H, 1)
                assert block_out[i % 2].shape[:-1] == block_softmax_lse[i % 2].shape[:-1]
                if i == 1:
                    pass
                # Output and log sum exp correction
                if i > 0:
                    sp_streams[i % 2].wait_event(correction_done)

                # Overlap output correction with next flash attn kernel
                if i == 0:
                    out = block_out[0]
                    softmax_lse = block_softmax_lse[0]
                elif i <= sp_rank:
                    _rescale_out_lse(out, block_out[i % 2], softmax_lse, block_softmax_lse[i % 2])
                else:
                    # Dropped the first half of q sequence
                    if is_packed:
                        _out, _lse = out[half_idx_back], softmax_lse[half_idx_back]
                    else:
                        _out, _lse = out[t // 2 :], softmax_lse[t // 2 :]
                    _rescale_out_lse(_out, block_out[i % 2], _lse, block_softmax_lse[i % 2])

                sp_streams[i % 2].record_event(correction_done)
        torch.cuda.current_stream().wait_event(correction_done)

        out = out.to(q.dtype)
        if not is_packed:
            out = out.view(b, sq, h, d)
            q, k, v = [x.view(b, sq, h, d) for x in (q, k, v)]  # (T, H, D) -> (B, Sq, H, D)
        softmax_lse = softmax_lse.squeeze(-1)

        ctx.sp_group = sp_group
        ctx.max_seqlen_q = max_seqlen
        ctx.max_seqlen_kv = max_seqlen_kv
        misc_kwargs["deterministic"] = deterministic
        del misc_kwargs["return_softmax"]
        ctx.misc_kwargs = misc_kwargs
        ctx.is_packed = is_packed
        indices = (half_idx_front, half_idx_back) if is_packed else tuple()
        ctx.save_for_backward(
            q,
            k,
            v,
            out,
            softmax_lse.transpose(0, 1).contiguous(),  # (T, H) -> (H, T)
            cu_seqlens_q,
            cu_seqlens_kv,
            *indices,
            *rng_states,
        )

        if return_softmax:
            return out, softmax_lse
        return out, None

    def backward(ctx, dout, _):
        """
        During backward, we accumulate q grads on each rank locally, but iterate kv and their grads
        over all ranks for accumulation.
        """
        (
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_kv,
        ) = ctx.saved_tensors[:7]
        is_packed = ctx.is_packed
        if is_packed:
            half_idx_front, half_idx_back = ctx.saved_tensors[7:9]
            rng_states = ctx.saved_tensors[9:]
            softmax_lse1 = softmax_lse[:, half_idx_back].contiguous()
        else:
            rng_states = ctx.saved_tensors[7:]
            softmax_lse1 = softmax_lse.chunk(2, dim=-1)[1].contiguous()  # Second half of seq

        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_kv = ctx.max_seqlen_kv
        misc_kwargs = ctx.misc_kwargs
        dout = dout.contiguous()
        del misc_kwargs["block_table"]

        assert (
            out.shape == dout.shape == q.shape
        ), f"out {out.shape} and dout {dout.shape} should have the same shape ({q.shape})."

        if is_packed:
            t, h, d = q.shape
        else:
            b, sq, h, d = q.shape
            t = b * sq
            q, k, v, out, dout = [x.view(t, h, d) for x in (q, k, v, out, dout)]

        # Sequence parallel args
        sp_group = ctx.sp_group
        sp_rank = dist.get_rank(sp_group)
        sp_size = dist.get_world_size(sp_group)
        kv_comm = RingComm(sp_group)
        dkv_comm = RingComm(sp_group)

        # Double comm buffers for sending and receiving kv
        kv_buffers = [torch.stack((k, v))]  # (2, T, H, D)
        kv_buffers.append(torch.empty_like(kv_buffers[0]))

        dq = None  # (T, H, D)
        # Intermediate outputs
        dq_block = torch.empty_like(q)  # (T, H, D)
        dk_block = torch.empty_like(k)  # (T, H, D)
        dv_block = torch.empty_like(v)  # (T, H, D)
        dkv_buffers = [torch.empty_like(kv, dtype=torch.float32) for kv in kv_buffers]  # (T, H, D)
        dkv_send = dkv_recv = None
        del k, v

        # NOTE: We avoid using two streams since it requires doubling dkv and kv buffers,
        # and backward is more communication intensive than forward
        for i in range(sp_size):
            kv_comm.wait()
            if i < sp_size - 1:
                # Send kv to next rank for backward
                kv_comm.send_recv(kv_buffers[i % 2], kv_buffers[(i + 1) % 2])

            if i == 0:
                # Backward with local kv
                k_, v_ = kv_buffers[i % 2]
                q_, dout_, out_ = q, dout, out
                dq_, dk_, dv_ = dq_block, dk_block, dv_block
                _flash_attn_backward(
                    dout_,
                    q_,
                    k_,
                    v_,
                    out_,
                    softmax_lse,
                    dq_,
                    dk_,
                    dv_,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    causal=True,
                    rng_state=rng_states[i],
                    **misc_kwargs,
                )
            elif i <= sp_rank:
                # Drop the second half of kv
                # (T, H, D) -> (T // 2, H, D)
                if is_packed:
                    k_, v_, dk_, dv_ = [x[half_idx_front] for x in (*kv_buffers[i % 2], dk_block, dv_block)]
                else:
                    k_, v_, dk_, dv_ = [x[: t // 2] for x in (*kv_buffers[i % 2], dk_block, dv_block)]
                dq_, q_, out_, dout_ = (dq_block, q, out, dout)

                _flash_attn_backward(
                    dout_,
                    q_,
                    k_,
                    v_,
                    out_,
                    softmax_lse,
                    dq_,
                    dk_,
                    dv_,
                    cu_seqlens_q,
                    cu_seqlens_kv // 2,
                    max_seqlen_q,
                    max_seqlen_kv // 2,
                    causal=False,
                    rng_state=rng_states[i],
                    **misc_kwargs,
                )

            else:
                # Drop the first half of q
                k_, v_ = kv_buffers[i % 2]
                dk_, dv_ = dk_block, dv_block
                if is_packed:
                    q_, dq_, out_, dout_ = [x[half_idx_back] for x in (q, dq_block, out, dout)]
                else:
                    q_, dq_, out_, dout_ = [x[t // 2 :] for x in (q, dq_block, out, dout)]
                _flash_attn_backward(
                    dout_,
                    q_,
                    k_,
                    v_,
                    out_,
                    softmax_lse1,
                    dq_,
                    dk_,
                    dv_,
                    cu_seqlens_q // 2,
                    cu_seqlens_kv,
                    max_seqlen_q // 2,
                    max_seqlen_kv,
                    causal=False,
                    rng_state=rng_states[i],
                    **misc_kwargs,
                )

            # Accumulate grads
            dkv_send = dkv_buffers[i % 2]
            dkv_recv = dkv_buffers[(i + 1) % 2]
            if i == 0:
                dq = dq_block.float()
                dkv_recv[0].copy_(dk_block)
                dkv_recv[1].copy_(dv_block)
            else:
                # Accumulate local dq
                if i <= sp_rank:
                    dq += dq_block  # (T, H, D)
                else:
                    if is_packed:
                        dq[half_idx_back] += dq_block[half_idx_back]
                    else:
                        dq[t // 2 :] += dq_block[t // 2 :]  # (T // 2, H, D)

                # Wait for mobile kv grad accumulators
                dkv_comm.wait()
                if i <= sp_rank:
                    # q blocks "surrounded" by kv blocks
                    if is_packed:
                        dkv_recv[0][half_idx_front] += dk_block[half_idx_front]
                        dkv_recv[1][half_idx_front] += dv_block[half_idx_front]
                    else:
                        dkv_recv[0][: t // 2] += dk_block[: t // 2]  # (T // 2, H, D)
                        dkv_recv[1][: t // 2] += dv_block[: t // 2]
                else:
                    # q blocks "surrounding" kv blocks
                    dkv_recv[0] += dk_block
                    dkv_recv[1] += dv_block

            dkv_comm.send_recv(send_tensor=dkv_recv, recv_tensor=dkv_send)
        dkv_comm.wait()
        dkv_recv = dkv_send

        dq, dk, dv = [x.to(q.dtype) for x in (dq, *dkv_recv)]
        if not is_packed:
            dq, dk, dv = [x.view(b, sq, h, d) for x in (dq, dk, dv)]
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def prepare_varlen_batch(
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        sp_group: dist.ProcessGroup,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Preprocess a batch of padded sequence by splitting input sequence by sp_size
        sequence-wise and packing them into one sequence. Updates the mask info accordingly.
        Args:
            inputs_embeds (torch.Tensor): Input embeddings. Shape should be [B, Sq, ...]
            attention_mask (torch.Tensor): Contains the mask [B, Sq]
            position_ids (Optional[torch.Tensor], optional): Position ids of shape [Sq]. Defaults to None.
        """
        _load_varlen_helpers()
        sp_size = dist.get_world_size(group=sp_group)
        sp_rank = dist.get_rank(group=sp_group)
        mask_info = {}
        mask_info["max_seqlen"], mask_info["cu_seqlens"] = get_pad_info(attention_mask, return_indices=False)

        # Unpad, split seq-wise, then pad back to (B, max_seqlen // sp_size)
        # Split mask to compute local nonzero position indices
        # (B, Sq) -> (B, max_seqlen // sp_size)
        attention_mask, inputs_embeds = split_varlen_zigzag(
            [attention_mask, inputs_embeds],
            mask_info["cu_seqlens"],
            sp_group,
            mask_info["max_seqlen"],
            is_2d=True,
        )
        mask_info["valid_indices"] = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        mask_info["max_seqlen"] //= sp_size
        mask_info["cu_seqlens"] //= sp_size
        mask_info["attention_mask_type"] = AttnMaskType.PADDED_CAUSAL

        if position_ids is not None:
            indices = torch.tensor([sp_rank, 2 * sp_size - sp_rank - 1], device=inputs_embeds.device)
            position_ids = (
                position_ids[: mask_info["max_seqlen"]].view(sp_size * 2, -1).index_select(0, indices).view(-1)
            )
        return inputs_embeds, position_ids, mask_info

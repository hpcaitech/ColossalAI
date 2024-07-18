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
def get_pad_info(padding_mask: torch.Tensor, invert: Optional[bool] = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """Get padding information from padding mask.

    Args:
        padding_mask (torch.Tensor): Padding mask tensor. Shape should be [B, Sq]
        invert (Optional[bool], optional): Whether to reverse the padding mask.
    Returns:
        max_seqlen_in_batch (int): Maximum sequence length in the batch.
        cu_seqlens (torch.Tensor): Shape [B+1]. Cumulative sequence lengths of the sequences in the batch.
        indices (torch.Tensor): Shape [B * Sq]. The indices of non-masked tokens from the flattened input sequence.
    """
    if invert:
        padding_mask = padding_mask.logical_not()
    seqlens_in_batch = padding_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return max_seqlen_in_batch, cu_seqlens, indices


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
    ) -> torch.Tensor:
        """Flash Attention function. It supports 4 mask type.
        1. custom mask: recv attention_mask
        2. padded mask: recv attention_mask, attention_mask_type, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, indices
        3. causal mask: recv attention_mask, attention_mask_type
        4. padded causal mask: recv attention_mask, attention_mask_type, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, indices

        Args:
            q (torch.Tensor): Query tensor. Shape should be [B, Heads, Sq, D]
            k (torch.Tensor): Key tensor. Shape should be [B, Heads, Sq, D]
            v (torch.Tensor): Value tensor. Shape should be [B, Heads, Sq, D]
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
            torch.Tensor: Output tensor. Shape should be [B, Heads, Sq, D]
        """
        # known issue: sdpa does not support attention mask which contains whole row of masked tokens, which leads to nan
        # this case is usaul when padding mask is used and self attention is performed
        # thus, we don't use sdpa when padding mask is used
        # sanity check
        if attention_mask is not None:
            assert torch.is_floating_point(attention_mask), "attention_mask should be a floating point tensor."
            if attention_mask_type in (
                AttnMaskType.CUSTOM,
                AttnMaskType.CAUSAL,
                AttnMaskType.PADDED,
                AttnMaskType.PADDED_CAUSAL,
            ):
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
        else:
            # if attention_mask is None, attention_mask_type should be the default value
            assert attention_mask_type == AttnMaskType.CUSTOM
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


def _load_flash_attn():
    """A light-weight loader to check whether flash-attn is installed.
    Can't use ColoAttention._dispatch_kernel because we mutate the backward pass
    """
    global _flash_attn_forward, _flash_attn_backward, _pad_input, _unpad_input
    if _flash_attn_forward is not None and _flash_attn_backward is not None:
        return
    from flash_attn.bert_padding import index_first_axis, pad_input
    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward as _flash_attn_backward
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward as _flash_attn_forward

    # Flash attn claims this is more efficient than torch's bool indexing due to avoiding
    # copying to other dims
    def unpad_input(hidden_states: torch.Tensor, indices: torch.Tensor):
        return index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices)

    _pad_input = pad_input
    _unpad_input = unpad_input


def ring_attn_p2p_comm(sp_rank, send_tensor, recv_tensor, send_dst, recv_src, sp_group):
    """No metadata as K, V sizes are fixed"""
    if sp_rank % 2 == 0:
        send_op = dist.P2POp(dist.isend, send_tensor, send_dst, group=sp_group)
        recv_op = dist.P2POp(dist.irecv, recv_tensor, recv_src, group=sp_group)
        send_recv_ops = [send_op, recv_op]
    else:
        recv_op = dist.P2POp(dist.irecv, recv_tensor, recv_src, group=sp_group)
        send_op = dist.P2POp(dist.isend, send_tensor, send_dst, group=sp_group)
        send_recv_ops = [recv_op, send_op]

    reqs = dist.batch_isend_irecv(send_recv_ops)
    return reqs


def _not_nan(x):
    return not (x.isnan().any() or x.isinf().any())


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
    stride_out_3,
    stride_out_per_step_0,
    stride_out_per_step_1,
    stride_out_per_step_2,
    stride_out_per_step_3,
    stride_lse_0,
    stride_lse_1,
    stride_lse_2,
    stride_lse_3,
    BLOCK_M: tl.constexpr,
):
    batch_id = tl.program_id(0)
    sq_id = tl.program_id(1)
    h_id = tl.program_id(2)
    d_id = tl.arange(0, D)

    out_idx = batch_id * stride_out_0 + sq_id * stride_out_1 + h_id * stride_out_2 + d_id * stride_out_3
    out_per_step_idx = (
        batch_id * stride_out_per_step_0
        + sq_id * stride_out_per_step_1
        + h_id * stride_out_per_step_2
        + d_id * stride_out_per_step_3
    )
    lse_idx = batch_id * stride_lse_0 + h_id * stride_lse_1 + sq_id * stride_lse_2 + tl.zeros(D) * stride_lse_3
    lse_step_idx = batch_id * stride_lse_0 + h_id * stride_lse_1 + sq_id * stride_lse_2 + tl.zeros(D) * stride_lse_3

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
    B, Sq, H, D = out.shape

    assert out.is_contiguous() and block_out.is_contiguous() and lse.is_contiguous() and block_lse.is_contiguous()

    # TODO: use 1d kernel?
    grid = lambda META: (triton.cdiv(Sq, META["BLOCK_M"]), B, H)
    _rescale_out_lse_kernel[grid](
        out,
        block_out,
        lse,
        block_lse,
        B,
        Sq,
        H,
        D,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        block_out.stride(0),
        block_out.stride(1),
        block_out.stride(2),
        block_out.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        lse.stride(3),
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
    # new_lse = max_scale + torch.log(1 + torch.exp(min_scale - max_scale))
    new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    new_block_lse = torch.exp(block_lse - new_lse)
    assert _not_nan(new_lse), new_lse
    # dist.barrier()
    assert _not_nan(new_block_lse), new_block_lse

    out.copy_(torch.exp(lse - new_lse) * out + new_block_lse * block_out)
    lse.copy_(new_lse)

    # block_out = block_out.float()
    # out.copy_(out - F.sigmoid(block_lse - lse) * (out - block_out))
    # lse.copy_(lse - F.logsigmoid(lse - block_lse))
    # assert not lse.isnan().any(), lse
    # assert not out.isnan().any(), out


#  From Megatron-LM. TODO: try Triton
# def flash_attn_out_correction(out, out_per_step, seq_dim, softmax_lse, softmax_lse_per_step):
#     softmax_lse_corrected_exp = torch.exp(softmax_lse_per_step - softmax_lse).movedim(2, seq_dim)
#     softmax_lse_corrected_exp = softmax_lse_corrected_exp.unsqueeze(-1)
#     out_corrected = out_per_step * softmax_lse_corrected_exp
#     out.add_(out_corrected)


# def flash_attn_softmax_lse_correction(softmax_lse, softmax_lse_per_step):
#     """
#     softmax_lse: (B, H, Sq)
#     softmax_lse_per_step: (B, H, Sq)
#     """
#     max_scale = torch.max(softmax_lse, softmax_lse_per_step)
#     min_scale = torch.min(softmax_lse, softmax_lse_per_step)
#     new_scale = max_scale + torch.log(1 + torch.exp(min_scale - max_scale))
#     softmax_lse.copy_(new_scale)


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
    MAX_SEQLEN: int = None
    ATTENTION_MASK: torch.Tensor = None  # [B, Sq]
    SUPPORTED_MASK_TYPES = (AttnMaskType.CAUSAL,)

    @staticmethod
    def attention(
        q,  # (B, H, Sq, D)
        k,
        v,
        sp_group,
        sp_stream,
        attention_mask,  # [B, Sq]
        attention_mask_type,
        cu_seq_lens_q=None,
        cu_seq_lens_kv=None,
        max_seq_len_q=None,
        max_seq_len_kv=None,
        dropout_p=0,
        softmax_scale=None,
        deterministic=False,
    ):
        assert (
            q.shape[2] == k.shape[2]
        ), "Q, K and V having different sequence lengths (inference or cross-attn)\
            is not supported yet in training."
        assert (
            attention_mask_type in RingAttention.SUPPORTED_MASK_TYPES
        ), f"Mask type {attention_mask_type} is not supported yet."

        b, h, sq, d = q.shape

        # Get sequence length info for varlen forward
        if attention_mask_type == AttnMaskType.CAUSAL:
            # All sequences share the same length
            cu_seqlens_q = cu_seqlens_kv = torch.arange(0, b * sq + 1, sq, device=q.device, dtype=torch.int32)
            max_seqlen_q = max_seqlen_kv = sq

        # "Packed" mode where sequences of different lengths are packed into [T, H, D]
        # TODO: This gets very complicated, as we need to ensure the each of the UNPADDED B
        # sequences are split evenly on each device in zigzag_split_batch.
        # (Ex: https://github.com/zhuzilin/ring-flash-attention/blob/49a50141bdce4e76418afe2051646c9a771fe867/test/test_zigzag_ring_flash_attn_varlen_func.py#L43)
        # Left some logics here; to be supported depending on demands.
        elif AttnMaskType.PADDED_CAUSAL:
            # TODO: compute cu_seqlens locally using valid_positions
            assert attention_mask is not None, "Padded attention requires inputing valid token positions!"
            # Sequences are padded to the same length in a training round, so reuse the mask info.
            if (
                RingAttention.ATTENTION_MASK
                and (RingAttention.ATTENTION_MASK.shape == attention_mask.shape)
                and (RingAttention.ATTENTION_MASK == attention_mask).all()
            ):
                cu_seqlens_q = cu_seqlens_kv = RingAttention.CU_SEQLENS
                max_seqlen_q = max_seqlen_kv = RingAttention.MAX_SEQLEN
            else:
                max_seqlen, cu_seqlens, valid_positions = get_pad_info(attention_mask)
                RingAttention.CU_SEQLENS = cu_seqlens
                RingAttention.MAX_SEQLEN = max_seqlen
                RingAttention.ATTENTION_MASK = attention_mask
                # To [T, H, D] where T is the number of non-zero tokens
                q, k, v = [_unpad_input(x, valid_positions) for x in (q, k, v)]

        out = RingAttention.apply(
            q,
            k,
            v,
            sp_group,
            sp_stream,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            dropout_p,
            softmax_scale,
            deterministic,
        )

        if attention_mask_type == AttnMaskType.PADDED_CAUSAL:
            # Pad and reshape back
            # [T, N, D] -> [B, H, Sq, D]
            out = _pad_input(out, valid_positions, b, sq)
        else:
            out = out.transpose(1, 2)  # [B, Sq, H, D] -> [B, H, Sq, D]

        return out

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sp_group: dist.ProcessGroup,
        sp_stream: torch.cuda.Stream,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        deterministic: bool = False,
    ):
        """
        Args:
            q (torch.Tensor): Query tensor. Shape should be [B, Heads, Sq, D]
            k (torch.Tensor): Key tensor. Shape should be [B, Heads, Sq, Sq, D]
            v (torch.Tensor): Value tensor. Shape should be [B, Heads, Sq, Sq, D]
            sp_group (Optional[dist.ProcessGroup]): Process group for sequence parallelism
            sp_tream (torch.cuda.Stream): An different stream for output correction.
            cu_seqlens_q (Optional[torch.Tensor], optional): The cumulative sequence lengths
                of the sequences in the batch, used to index into q.
                Shape should be [B+1]. Defaults to None.
            cu_seqlens_kv (Optional[torch.Tensor], optional): The cumulative sequence lengths
                of the sequences in the batch, used to index into kv.
                Shape should be [B+1]. Defaults to None.
            max_seqlen_q (Optional[int], optional): Maximum query sequence length in the batch. Defaults to None.
            max_seqlen_kv (Optional[int], optional): Maximum key/value sequence length in the batch. Defaults to None.
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.
            softmax_scale (Optional[float], optional): Scaling factor applied prior to softmax. Defaults to None.
            deterministic (bool, optional): Whether to force deterministic backward pass. See https://github.com/Dao-AILab/flash-attention/issues/349
        Returns:
            torch.Tensor: Output tensor. Shape should be [B, Heads, Sq, D]
        """
        try:
            _load_flash_attn()
        except Exception as e:
            raise RuntimeError(
                f"Ring attention requires Flash Attention, but import failed. You can install it via 'pip install flash-attn --no-build-isolation'"
            ) from e

        misc_kwargs = {
            "window_size": (-1, -1),
            "alibi_slopes": None,
            "softmax_scale": q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale,
            "dropout_p": dropout_p,
            "block_table": None,
        }

        b, h, sq, d = q.shape
        # (B, H, Sq, D) -> (B, Sq, H, D)
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]

        sp_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)
        sp_global_ranks = dist.get_process_group_ranks(sp_group)
        send_dst = sp_global_ranks[(sp_rank + 1) % sp_size]
        recv_src = sp_global_ranks[(sp_rank - 1) % sp_size]

        # Pre-allocate double buffer for overlapping and receiving next step's inputs
        kv_buffers = [torch.stack((k, v))]  # (2, B, Sq, H, D)
        kv_buffers.append(torch.empty_like(kv_buffers[0]))

        # outputs
        out = None
        block_out = [None, None]
        softmax_lse = [None, None]
        block_softmax_lse = [None, None]  # log sum exp, the denominator of softmax in attention
        rng_states = [None for _ in range(sp_size)]
        sp_streams = [torch.cuda.current_stream(), sp_stream]
        correction_done = torch.cuda.Event()
        # Overlap output correction with next flash attn
        p2p_reqs = [[], []]
        for i in range(sp_size):
            # Wait for current kv from prev rank
            with torch.cuda.stream(sp_streams[i % 2]):
                for req in p2p_reqs[(i + 1) % 2]:
                    req.wait()
                assert _not_nan(kv_buffers[i % 2]), kv_buffers[i % 2]

                if i < sp_size - 1:
                    p2p_reqs[i % 2] = ring_attn_p2p_comm(
                        sp_rank,
                        kv_buffers[i % 2],  # send current kv to next rank
                        kv_buffers[(i + 1) % 2],  # recv from prev rank
                        send_dst,
                        recv_src,
                        sp_group,
                    )

                    if i == 0:
                        # Compute with local KV; no mask
                        q_block = q.view(b * sq, h, d)
                        # NOTE: clone to avoid buffer being overwritten by the next p2p comm call
                        kv_block = kv_buffers[i % 2].view(2, b * sq, h, d).clone()
                        (
                            _,
                            _,
                            _,
                            _,
                            block_out[i % 2],
                            block_softmax_lse[i % 2],
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
                            # Seems that the flash attn interface requires the dropout > 0 here
                            # (see https://github.com/Dao-AILab/flash-attention/issues/871)
                            # but returns softmax_lse anyway?
                            return_softmax=False,
                            **misc_kwargs,
                        )
                    elif i <= sp_rank:
                        # Received the "surrounding" kv chunks
                        # Drop the second half of received kv
                        q_block = q.view(b * sq, h, d)
                        kv_block = kv_buffers[i % 2]
                        # (2, B * Sq // 2, H, D)
                        kv_block = kv_block.view(2, b * sq, h, d)[:, : b * sq // 2].clone()
                        assert _not_nan(kv_block), f"rank {sp_rank} step {i} kv_block {kv_block}"
                        # actual_lse = (q_block.flatten(start_dim=1) @ kv_block[0].movedim(0, -1).flatten(end_dim=-2)).exp().sum(dim=-1).log()
                        (
                            _,
                            _,
                            _,
                            _,
                            block_out[i % 2],  # (B, Sq, H, D)
                            block_softmax_lse[i % 2],  # (B, H, Sq)
                            _,
                            rng_states[i],
                        ) = _flash_attn_forward(
                            q_block,
                            kv_block[0],
                            kv_block[1],
                            cu_seqlens_q,
                            cu_seqlens_kv // 2,
                            max_seqlen_q,
                            max_seqlen_kv // 2,
                            causal=False,
                            return_softmax=False,
                            **misc_kwargs,
                        )
                    else:
                        # Received the inner kv chunks
                        # Drop the first half of q
                        q_block = q.view(b * sq, h, d)[b * sq // 2 :]
                        kv_block = kv_buffers[i % 2].view(2, b * sq, h, d).clone()
                        assert _not_nan(kv_block), f"rank {sp_rank} step {i} kv_block {kv_block}"
                        # actual_lse = (q_block.flatten(start_dim=1) @ kv_block[0].movedim(0, -1).flatten(end_dim=-2)).exp().sum(dim=-1).log()

                        (
                            _,
                            _,
                            _,
                            _,
                            block_out[i % 2],  # (B, Sq // 2, H, D)
                            block_softmax_lse[i % 2],  # (B, H, Sq // 2)
                            _,
                            rng_states[i],
                        ) = _flash_attn_forward(
                            q_block,
                            kv_block[0],
                            kv_block[1],
                            cu_seqlens_q // 2,
                            cu_seqlens_kv,
                            max_seqlen_q // 2,
                            max_seqlen_kv,
                            causal=False,
                            return_softmax=False,
                            **misc_kwargs,
                        )
                    # Output and log sum exp correction
                    if i > 1:
                        sp_streams[i % 2].wait_event(correction_done)

                    block_out[i % 2] = block_out[i % 2].view(b, block_out[i % 2].shape[0] // b, h, d)
                    block_softmax_lse[i % 2] = (
                        block_softmax_lse[i % 2].transpose(1, 2).contiguous().unsqueeze(-1).float()
                    )

                    assert block_out[i % 2].shape[:-1] == block_softmax_lse[i % 2].shape[:-1]
                    assert _not_nan(
                        block_softmax_lse[i % 2]
                    ), f"rank {sp_rank} step {i} softmax_lse is nan: {block_softmax_lse[i % 2]}"

                    # Overlap output correction with next flash attn kernel
                    if i == 0:
                        out = block_out[0]
                        softmax_lse = block_softmax_lse[0]
                    elif i <= sp_rank:
                        _rescale_out_lse(out, block_out[i % 2], softmax_lse, block_softmax_lse[i % 2])
                    else:
                        # Dropped the first half of q sequence
                        _rescale_out_lse(
                            out[:, sq // 2 :], block_out[i % 2], softmax_lse[:, sq // 2 :], block_softmax_lse[i % 2]
                        )
                    sp_streams[i % 2].record_event(correction_done)

            torch.cuda.current_stream().wait_event(correction_done)

            out = out.view(b, sq, h, d).to(q.dtype)  # (B, Sq, H, D)
            q, k, v = [x.view(b, sq, h, d) for x in (q, k, v)]  # (B * Sq, H, D) -> (B, Sq, H, D)
            ctx.save_for_backward(
                q,
                k,
                v,
                out,
                softmax_lse,
                cu_seqlens_q,
                cu_seqlens_kv,
                *rng_states,
            )
            ctx.sp_group = sp_group
            ctx.sp_global_ranks = sp_global_ranks
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_kv = max_seqlen_kv
            misc_kwargs["deterministic"] = deterministic
            ctx.misc_kwargs = misc_kwargs

            return out.transpose(1, 2)  # Back to common layout (B, H, Sq, D) for compatibility

    def backward(ctx, dout):
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
        rng_states = ctx.saved_tensors[7:]
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_kv = ctx.max_seqlen_kv
        misc_kwargs = ctx.misc_kwargs
        del misc_kwargs["block_table"]

        dout = dout.transpose(1, 2).contiguous()  # (B, Sq, H, D)
        b, sq, h, d = q.shape
        assert (
            out.shape == dout.shape == (b, sq, h, d)
        ), f"out {out.shape} and dout {dout.shape} should have shape ({b}, {sq}, {h}, {d}) instead"

        # Sequence parallel args
        sp_group = ctx.sp_group
        sp_rank = dist.get_rank(sp_group)
        sp_size = dist.get_world_size(sp_group)
        sp_global_ranks = ctx.sp_global_ranks
        send_dst = sp_global_ranks[(sp_rank + 1) % len(sp_global_ranks)]
        recv_src = sp_global_ranks[(sp_rank - 1) % len(sp_global_ranks)]

        # Double comm buffers for sending and receiving kv
        kv_buffers = [torch.stack((k, v))]  # (B, Sq, H, D)
        kv_buffers.append(torch.empty_like(kv_buffers[0]))
        dkv_buffers = [torch.empty_like(kv_buffers[0]) for _ in range(2)]

        dq = torch.empty_like(q)  # (B, Sq, H, D)
        # Intermediate outputs
        dq_block = torch.empty_like(q)  # (B, Sq, H, D)
        dk_block = torch.empty_like(q)  # (B, Sq, H, D)
        dv_block = torch.empty_like(q)  # (B, Sq, H, D)
        del k, v

        kv_reqs = []
        dkv_reqs = []
        # NOTE: We avoid using two streams since it requires doubling dkv and kv buffers,
        # and backward is more communication intensive than forward
        for i in range(sp_size):
            for req in kv_reqs:
                req.wait()
            if i < sp_size - 1:
                # Send kv to next rank for backward
                kv_reqs = ring_attn_p2p_comm(
                    sp_rank,
                    send_tensor=kv_buffers[i % 2],
                    recv_tensor=kv_buffers[(i + 1) % 2],
                    send_dst=send_dst,
                    recv_src=recv_src,
                    sp_group=sp_group,
                )
            if i == 0:
                # Backward with local kv
                k_, v_ = [x.view(b * sq, h, d) for x in kv_buffers[i % 2]]
                q_, dout_, out_ = [x.view(b * sq, h, d) for x in (q, dout, out)]
                dq_, dk_, dv_ = (x.view(b * sq, h, d) for x in (dq_block, dk_block, dv_block))

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
                # Drop the first half of kv
                # (B, Sq, H, D) -> (B * Sq // 2, H, D)
                k_, v_, dk_, dv_ = [
                    x.view(b * sq, h, d)[: b * sq // 2] for x in (*kv_buffers[i % 2], dk_block, dv_block)
                ]
                # dk_, dv_ = (x[:, 1].view(b * sq // 2, h, d) for x in (dk_block, dv_block))
                dq_, q_, out_, dout_ = [x.view(b * sq, h, d) for x in (dq_block, q, out, dout)]

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
                k_, v_ = [x.view(b * sq, h, d) for x in kv_buffers[i % 2]]
                dk_, dv_ = (x.view(b * sq, h, d) for x in (dk_block, dv_block))
                dq_, q_, out_, dout_ = [x.view(b * sq, h, d)[b * sq // 2 :] for x in (dq_block, q, out, dout)]

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
                    cu_seqlens_q // 2,
                    cu_seqlens_kv,
                    max_seqlen_q // 2,
                    max_seqlen_kv,
                    causal=False,
                    rng_state=rng_states[i],
                    **misc_kwargs,
                )

            # Accumulate grads
            if i == 0:
                # TODO: use float() if precision goes wrong
                dq = dq_block
                dk_recv = dkv_buffers[(i + 1) % 2][0] = dk_block.clone()
                dv_recv = dkv_buffers[(i + 1) % 2][1] = dv_block.clone()
            else:
                # Accumulate local dq
                if i <= sp_rank:
                    dq += dq_block  # (B, Sq, H, D)
                else:
                    dq_block = dq_block[:, sq // 2 :]  # (B, Sq // 2, H, D)
                    dq[:, sq // 2 :] += dq_block

                # Wait for mobile kv grad accumulators
                for req in dkv_reqs:
                    req.wait()

                if i <= sp_rank:
                    # q blocks "surrounded" by kv blocks
                    dk_recv = dkv_buffers[(i + 1) % 2][0]
                    dv_recv = dkv_buffers[(i + 1) % 2][1]
                    dk_recv[:, : sq // 2] += dk_block[:, : sq // 2]  # (B, Sq // 2, H, D)
                    dv_recv[:, : sq // 2] += dv_block[:, : sq // 2]
                else:
                    # q blocks "surrounding" kv blocks; full kv grads
                    dk_recv = dkv_buffers[(i + 1) % 2][0]
                    dv_recv = dkv_buffers[(i + 1) % 2][1]
                    dk_recv += dk_block
                    dv_recv += dv_block

            if i < sp_size - 1:
                dkv_reqs = ring_attn_p2p_comm(
                    sp_rank,
                    send_tensor=dkv_buffers[(i + 1) % 2],
                    recv_tensor=dkv_buffers[i % 2],
                    send_dst=send_dst,
                    recv_src=recv_src,
                    sp_group=sp_group,
                )

        dq, dk, dv = [x.view(b, sq, h, d).transpose(1, 2) for x in (dq, dk_recv, dv_recv)]
        return (dq, dk, dv, None, None, None, None, None, None, None, None, None)

from enum import Enum
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.distributed
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from packaging import version

from colossalai.kernel.kernel_loader import (
    FlashAttentionDaoLoader,
    FlashAttentionForFloatAndCustomMaskLoader,
    FlashAttentionLoader,
    FlashAttentionWithCustomMaskLoader,
    KernelLoader,
)
from colossalai.logging import get_dist_logger

from .utils import RingComm, get_half_index, split_varlen_zigzag

MEMORY_BOUND = 10 * 1e9

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
        mask (torch.Tensor): Mask tensor. Shape should be [B, 1, Sq, Skv]

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
        padding_mask (torch.Tensor): Padding mask tensor. Shape should be [B, Skv]
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
    _flash_kernel_dispatch: Optional[Dict[torch.dtype, Dict[Optional[AttnMaskType], Callable]]] = None

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
        if ColoAttention._flash_kernel_dispatch is None:
            ColoAttention._flash_kernel_dispatch = FlashAttentionDaoLoader()

    @staticmethod
    def _dispatch_kernel(dtype: torch.dtype, mask_type: Optional[AttnMaskType], size) -> Callable:
        ColoAttention._init_kernels_dispatch()
        if (
            dtype not in ColoAttention._kernel_dispatch_map
            or mask_type not in ColoAttention._kernel_dispatch_map[dtype]
        ):
            raise ValueError(
                "FlashAttention kernel is not available for dtype {} and mask_type {}".format(dtype, mask_type)
            )

        if size >= MEMORY_BOUND:
            if isinstance(ColoAttention._flash_kernel_dispatch, KernelLoader):
                ColoAttention._flash_kernel_dispatch = ColoAttention._flash_kernel_dispatch.load()
        # lazy load
        if isinstance(ColoAttention._kernel_dispatch_map[dtype][mask_type], KernelLoader):
            ColoAttention._kernel_dispatch_map[dtype][mask_type] = ColoAttention._kernel_dispatch_map[dtype][
                mask_type
            ].load()

        if size >= MEMORY_BOUND and mask_type in (AttnMaskType.PADDED_CAUSAL, AttnMaskType.CAUSAL):
            return ColoAttention._flash_kernel_dispatch
        else:
            return ColoAttention._kernel_dispatch_map[dtype][mask_type]

    @staticmethod
    def prepare_attn_kwargs(
        shape_4d: Tuple[int],
        dtype: torch.dtype,
        device: torch.device,
        q_padding_mask: Optional[torch.Tensor] = None,
        kv_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        invert: bool = True,
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
            invert_mask (bool, optional): Whether to invert the mask. Defaults to True.
        Returns:
            Dict[str, torch.Tensor]: Dictionary of keyword arguments for attention function.
        """
        if q_padding_mask is None and not is_causal:
            return {}
        assert len(shape_4d) == 4 and shape_4d[1] == 1
        b, _, s_q, s_kv = shape_4d
        element_size = torch.tensor([], dtype=dtype).element_size()
        memory_size = s_q * s_kv * element_size
        outputs = {}
        if (q_padding_mask is None or q_padding_mask.bool().all()) and (
            kv_padding_mask is None or kv_padding_mask.bool().all()
        ):
            # no padding
            assert is_causal
            outputs["attention_mask_type"] = AttnMaskType.CAUSAL
            if memory_size < MEMORY_BOUND:
                attention_mask = torch.ones(s_q, s_kv, dtype=dtype, device=device)
                if s_q != 1:
                    attention_mask.tril_(diagonal=0)
                attention_mask = attention_mask.expand(b, s_q, s_kv)
            else:
                attention_mask = torch.empty((0,), dtype=dtype, device=device)
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
            ), f"Padding mask shape {kv_padding_mask.shape} should align with shape 4d ({b}, {s_kv})"
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
                if memory_size < MEMORY_BOUND:
                    if s_q != 1:
                        attention_mask = kv_padding_mask[:, None, :].expand(b, s_q, s_kv).to(dtype=dtype, device=device)
                        attention_mask = attention_mask * attention_mask.new_ones(s_q, s_kv).tril(diagonal=0)
                else:
                    attention_mask = torch.empty((0,), dtype=dtype, device=device)
            else:
                outputs["attention_mask_type"] = AttnMaskType.PADDED
                if memory_size < MEMORY_BOUND:
                    attention_mask = kv_padding_mask[:, None, :].expand(b, s_q, s_kv).to(dtype=dtype, device=device)

        if invert:
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
            k (torch.Tensor): Key tensor. Shape should be [B, nHeads, Skv, D]
            v (torch.Tensor): Value tensor. Shape should be [B, nHeads, Skv, D]
            attention_mask (Optional[torch.Tensor], optional): Attention mask tensor. Shape should be [B, 1, Sq, Skv]. Defaults to None.
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
        else:
            # if attention_mask is None, attention_mask_type should be the default value
            assert attention_mask_type == AttnMaskType.CUSTOM

        # kernel dispatch
        b, _, s_q, _ = q.shape
        b, _, s_kv, _ = v.shape
        element_size = torch.tensor([], dtype=q.dtype).element_size()
        memory_size = s_q * s_kv * element_size
        mask_type = attention_mask_type if attention_mask is not None else None
        attn_func = ColoAttention._dispatch_kernel(q.dtype, mask_type, memory_size)
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


# NOTE: This can cause spawned processes to hang on exit
# with python 3.9
@torch.compile()
def _rescale_out_lse(out, block_out, lse, block_lse):
    """
    Compute the new attention denominator:
        exp(lse) + exp(block_lse) = exp(max_scale) * (exp(min_scale - max_scale) + 1)
    Args:
        out: (T, H, D)
        block_out: (T, H, D)
        lse: (H, T, 1)
        block_lse: (H, T, 1)
    """

    # min_scale = torch.min(lse, block_lse)
    # max_scale = torch.max(lse, block_lse)
    # new_lse = max_scale + torch.log(1 + torch.exp(min_scale - max_scale))

    # NOTE: directly assigning to .data here is buggy
    # probably due to casting dtypes/strides
    new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))

    new_block_lse = torch.exp(block_lse - new_lse)
    out = (torch.exp(lse - new_lse) * out + new_block_lse * block_out).to(out)
    lse = new_lse

    # Equivalent to the above
    # See https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    # out = (out - F.sigmoid(block_lse - lse) * (out - block_out))
    # lse = (lse - F.logsigmoid(lse - block_lse))
    return out, lse


class RingAttention(torch.autograd.Function):
    """Implements the Ring Attention from `Ring Attention with Blockwise Transformers for Near-Infinite Context`
    (https://arxiv.org/abs/2310.01889).
    For load-balancing we adopted the "zigzag" attention scheme from https://github.com/zhuzilin/ring-flash-attention/tree/main
    For portable integration with more models, we don't follow the spirit of "block-wise FNN" in the original paper,
    which requires fusing FFN with the Flash Attention kernel/function (see https://arxiv.org/pdf/2305.19370;
    implemented in Jax and not optimized).
    We adopt the double ring topology from LoongTrain (https://arxiv.org/pdf/2406.18485) to fully utilize available
    NICs on each node, by computing attention within a inner ring first and then sending all KVs to the next
    ring at once.
    """

    # Globle cache to avoid recomputation for same-lengthed sequences
    CU_SEQLENS: torch.Tensor = None  # [B+1]
    TOTAL_SEQLEN: int = None
    HALF_INDICES: Tuple = None
    SUPPORTED_MASK_TYPES = (AttnMaskType.CAUSAL, AttnMaskType.PADDED_CAUSAL)
    ATTN_DONE: torch.cuda.Event = None
    SP_STREAM: torch.cuda.Stream = None
    SP_GROUP: dist.ProcessGroup = None

    # NOTE: Duplicating PGs for concurrent NCCL streams is a risky hack -- while it may increase throughput,
    # both PyTorch and NCCL warn against this. (https://github.com/pytorch/pytorch/commit/2dbe5cb979f674f0052a8eea1f7b6c3c0ba441d7)
    # LoongTrain's original double ring impl. uses concurrent PGs
    # (https://github.com/InternLM/InternEvo/blob/e52f2ffc9acf818e8f2b1f97dfc69ceb2f06e154/internlm/model/ops/ring_flash_attn/zigzag_ring_flash_attn_with_sliding_window.py#L192)
    # but I confirmed with Pytorch developers this can cause obscure "Software caused connection abort" errors.
    # (https://github.com/pytorch/pytorch/issues/132852)
    # NOTE: In general, a smarter idea is put as many P2P calls as possible into one `batch_isend_irecv`.
    INNER_RING_GROUP: dist.ProcessGroup = None
    # INNER_RING_GROUP_COPY: dist.ProcessGroup = None
    INTER_RING_GROUP: dist.ProcessGroup = None
    # INTER_RING_GROUP_COPY: dist.ProcessGroup = None

    @staticmethod
    def get_double_ring_groups(sp_axis, pg_mesh, inner_ring_size=None):
        """
        Get 2D ring groups for the given process group. Generally, to avoid congestion, the inner ring size
        shouldn't be larger than the number of NICs on each node.
        Args:
            sp_group (dist.ProcessGroup): Process group for sequence parallelism
            inner_ring_size (Optional[int], optional): Inner ring size. Defaults to None.
        Returns:
            Tuple[dist.ProcessGroup, dist.ProcessGroup]: Inner-ring process group and inter-ring process group.
        """
        assert pg_mesh is not None, f"Error: The pg mesh is None! please check the process group initialization."

        sp_group = pg_mesh.get_group_along_axis(sp_axis)
        sp_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)

        assert inner_ring_size is not None

        assert (
            inner_ring_size <= sp_size and sp_size % inner_ring_size == 0
        ), f"Error: sp_size {sp_size} should be divisible by inner_ring_size {inner_ring_size}"

        if inner_ring_size == sp_size:
            return sp_group, sp_group
        assert (
            sp_size % inner_ring_size == 0
        ), f"sp_size {sp_size} should be divisible by inner_ring_size {inner_ring_size}"
        logger = get_dist_logger()
        logger.info(
            f"Using 2D Ring Attention with inner ring size {inner_ring_size} to maximze NIC util for inter-node comm. Cross your fingers for speed-ups!",
            ranks=[0],
        )
        num_rings = sp_size // inner_ring_size
        inner_ring_group = None
        inter_ring_group = None

        # Create inner ring groups
        for i in range(inner_ring_size):
            ranks = list(range(i * inner_ring_size, (i + 1) * inner_ring_size))
            group = pg_mesh.get_group_along_axis(sp_axis, ranks)
            if sp_rank in ranks:
                inner_ring_group = group

        # Create inter ring groups
        for i in range(num_rings):
            ranks = list(range(i, sp_size, num_rings))
            group = pg_mesh.get_group_along_axis(sp_axis, ranks)
            if sp_rank in ranks:
                inter_ring_group = group

        return inner_ring_group, inter_ring_group

    @staticmethod
    def attention(
        q,  # (B, H, Sq, D)
        k,
        v,
        sp_axis,
        attention_mask_type,
        cu_seqlens=None,
        max_seqlen=None,
        valid_indices=None,
        dropout_p=0.0,
        softmax_scale=None,
        deterministic=False,
        return_softmax=False,
        inner_ring_size=None,
        pg_mesh=None,
        **kwargs,
    ):
        """
        Ring Attention forward pass supporting variable-length sequences. When using varlen mode,
        each sequence in the batch should have length divisible by sp_size * 2.

        Args:
            q (torch.Tensor): Query tensor. Shape should be [B, nHeads, Sq, D]
            k (torch.Tensor): Key tensor. Shape should be [B, nHeads, Sq, Sq, D]
            v (torch.Tensor): Value tensor. Shape should be [B, nHeads, Sq, Sq, D]
            sp_axis (Optional[int]): Sp axis for the global pg mesh.
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
            inner_ring_size (Optional[int], optional): Inner ring size of the 2D ring. By default use a heuristic to decide.

        Returns:
            out: Output tensor of shape [B, nHeads, Sq, D] or [T, nHeads, D] if pad_output is False.
            softmax_lse: (if return_softmax is True) Softmax denominator (logsumexp).
                Shape should be [total_q_seqlen, nHeads]
        """
        # Check input args
        _load_flash_attn()
        if RingAttention.ATTN_DONE is None:
            RingAttention.ATTN_DONE = torch.cuda.Event()
        if RingAttention.SP_STREAM is None:
            RingAttention.SP_STREAM = torch.cuda.Stream()
        assert (
            q.shape[2] == k.shape[2]
        ), "Q, K and V having different sequence lengths (inference or cross-attn)\
            is not supported yet in training."
        assert (
            attention_mask_type in RingAttention.SUPPORTED_MASK_TYPES
        ), f"Mask type {attention_mask_type} is not supported yet."

        assert pg_mesh is not None, f"Error: The pg mesh is None! please check the process group initialization."

        clone_pg = lambda pg: dist.new_group(dist.get_process_group_ranks(pg))
        sp_group = pg_mesh.get_group_along_axis(sp_axis)
        if inner_ring_size != None:
            RingAttention.SP_GROUP = sp_group
            inner_ring_group, inter_ring_group = RingAttention.get_double_ring_groups(sp_axis, pg_mesh, inner_ring_size)
            RingAttention.INNER_RING_GROUP = inner_ring_group
            RingAttention.INTER_RING_GROUP = inter_ring_group
        else:
            inner_ring_group = RingAttention.INNER_RING_GROUP
            inter_ring_group = RingAttention.INTER_RING_GROUP

        # (B, H, Sq, D) -> (B, Sq, H, D)
        q, k, v = [x.transpose(1, 2).contiguous() for x in (q, k, v)]
        pad_output = q.dim() == 4

        # Get sequence length info for varlen forward
        if attention_mask_type == AttnMaskType.CAUSAL:
            # All sequences share the same length
            b, sq, h, d = q.shape
            max_seqlen = sq
            # Cache to avoid recreation for a single sequence
            if sq * b == RingAttention.TOTAL_SEQLEN:
                cu_seqlens = RingAttention.CU_SEQLENS
            else:
                cu_seqlens = torch.arange(0, b * sq + 1, sq, device=q.device, dtype=torch.int32)
                RingAttention.TOTAL_SEQLEN = b * sq

        # "Packed" mode where sequences of different lengths are packed into [total_q_seqlen, H, D]
        elif attention_mask_type == AttnMaskType.PADDED_CAUSAL:
            assert (
                cu_seqlens is not None and max_seqlen is not None and valid_indices is not None
            ), "Packed mode requires pre-computed cu_seqlens and max_seq_len."
            if pad_output:
                b, sq, h, d = q.shape
                q, k, v = [_unpad_input(x, valid_indices) for x in (q, k, v)]

        out, softmax_lse = RingAttention.apply(
            q,
            k,
            v,
            sp_group,
            RingAttention.SP_STREAM,
            cu_seqlens,
            max_seqlen,
            dropout_p,
            softmax_scale,
            deterministic,
            return_softmax,
            attention_mask_type == AttnMaskType.PADDED_CAUSAL,
            inner_ring_group,
            inter_ring_group,
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
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        deterministic: Optional[bool] = False,
        return_softmax: Optional[bool] = False,
        is_packed: Optional[bool] = False,
        inner_ring_group: Optional[dist.ProcessGroup] = None,
        inter_ring_group: Optional[dist.ProcessGroup] = None,
    ):
        """
        Forward supporting both packed (varlen) and batched(fixed length, no padding) sequences.
        No separate version for batched seq (hard to maintain), which incurs
        some overhead in sequence splitting due to python for loops.
        Uses two CUDA streams to overlap softmax denominator correction with next flash attn
        (see comments below).
        """
        cu_seqlens_q = cu_seqlens_kv = cu_seqlens
        max_seqlen_q = max_seqlen_kv = max_seqlen
        cu_seqlens_half = cu_seqlens // 2
        max_seqlen_half = max_seqlen // 2
        misc_kwargs = {
            "alibi_slopes": None,
            "softmax_scale": q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale,
            "dropout_p": dropout_p,
            "block_table": None,
            "softcap": 0.0,
            "return_softmax": False,
        }
        import flash_attn

        if version.parse(flash_attn.__version__) > version.parse("2.6.3"):
            misc_kwargs["window_size_left"] = -1
            misc_kwargs["window_size_right"] = -1
        else:
            misc_kwargs["window_size"] = (-1, -1)

        if (
            RingAttention.HALF_INDICES is not None
            and cu_seqlens.shape == RingAttention.CU_SEQLENS.shape
            and (cu_seqlens == RingAttention.CU_SEQLENS).all()
        ):
            half_idx_front, half_idx_back = RingAttention.HALF_INDICES
        else:
            half_idx_front = get_half_index(cu_seqlens, front=True)
            half_idx_back = get_half_index(cu_seqlens, front=False)
            RingAttention.HALF_INDICES = (half_idx_front, half_idx_back)
            RingAttention.CU_SEQLENS = cu_seqlens

        if is_packed:
            t, h, d = q.shape
        else:
            b, sq, h, d = q.shape
            t = b * sq
            # Be careful about GQA/MQA in reshape
            q, k, v = [x.view(t, *x.shape[-2:]) for x in (q, k, v)]

        if inner_ring_group is None or inter_ring_group is None:
            # Use one ring if not specified
            inner_ring_group = inter_ring_group = sp_group

        sp_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)

        # Create communicators corresponding to two CUDA streams
        local_kv_comms = [RingComm(inner_ring_group) for _ in range(2)]
        inter_ring_comm = RingComm(inter_ring_group)
        local_sp_size = dist.get_world_size(inner_ring_group)
        local_sp_rank = dist.get_rank(inner_ring_group)
        inter_ring_rank = dist.get_rank(inter_ring_group) if inter_ring_group is not sp_group else 0
        num_rings = dist.get_world_size(inter_ring_group) if inter_ring_group is not sp_group else 1

        # Any type of indexing(but not slicing) copies to a new contiguous tensor,
        # so only do it once
        if sp_rank != sp_size - 1:
            q1 = q[half_idx_back]

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

        # Helper to pass args to FA
        def _forward(q, k, v, causal):
            if version.parse(flash_attn.__version__) > version.parse("2.6.3"):
                (out, softmax_lse, S_dmask, rng_state) = _flash_attn_forward(
                    q,
                    k,
                    v,
                    cu_seqlens_q if q.shape[0] == t else cu_seqlens_half,
                    cu_seqlens_kv if k.shape[0] == t else cu_seqlens_half,
                    max_seqlen_q if q.shape[0] == t else max_seqlen_half,
                    max_seqlen_kv if k.shape[0] == t else max_seqlen_half,
                    causal=causal,
                    **misc_kwargs,
                )
            else:
                (
                    _,
                    _,
                    _,
                    _,
                    out,
                    softmax_lse,
                    _,
                    rng_state,
                ) = _flash_attn_forward(
                    q,
                    k,
                    v,
                    cu_seqlens_q if q.shape[0] == t else cu_seqlens_half,
                    cu_seqlens_kv if k.shape[0] == t else cu_seqlens_half,
                    max_seqlen_q if q.shape[0] == t else max_seqlen_half,
                    max_seqlen_kv if k.shape[0] == t else max_seqlen_half,
                    causal=causal,
                    **misc_kwargs,
                )
            return out, softmax_lse, rng_state

        def _kv_comm(i):
            # Avoid overwriting attn input when it shares mem with buffer
            if not RingAttention.ATTN_DONE.query():
                kv_buffers[(i + 1) % 2] = torch.empty_like(kv_buffers[i % 2])
            if i < local_sp_size - 1:
                local_kv_comms[i % 2].send_recv(kv_buffers[i % 2], kv_buffers[(i + 1) % 2])

        # Forward within a node
        def _local_ring_forward():
            # (Hopefully) overlap output correction with next flash attn
            for i in range(local_sp_size):
                with torch.cuda.stream(sp_streams[i % 2]):
                    # Wait for current kv from prev rank
                    # NOTE: waiting outside the current stream will NOT correctly synchronize.
                    if i > 0:
                        local_kv_comms[(i + 1) % 2].wait()

                    # Prefetch
                    if i == 0:
                        _kv_comm(i)

                    if i == 0:
                        # Compute with local KV; no mask
                        kv_block = kv_buffers[0]
                        q_block = q
                        (block_out[i % 2], block_softmax_lse[i % 2], rng_states[i]) = _forward(  # (T, H, D)  # (H, T)
                            q_block, kv_block[0], kv_block[1], causal=True
                        )
                    elif i <= local_sp_rank:
                        # Received the "surrounding" kv chunks
                        # Drop the second half of received kv
                        # (2, t // 2, H, D)
                        kv_block = kv_buffers[i % 2][:, half_idx_front]
                        q_block = q
                        (
                            block_out[i % 2],  # (T, H, D)
                            block_softmax_lse[i % 2],  # (H, T)
                            rng_states[i],
                        ) = _forward(q_block, kv_block[0], kv_block[1], causal=False)
                    else:
                        # Received the inner kv chunks
                        # Drop the first half of q
                        kv_block = kv_buffers[i % 2]
                        q_block = q1
                        (
                            block_out[i % 2],  # (T, H, D)
                            block_softmax_lse[i % 2],  # (H, T)
                            rng_states[i],
                        ) = _forward(q_block, kv_block[0], kv_block[1], causal=False)
                    RingAttention.ATTN_DONE.record()
                    # Pipeline the next KV comm with output correction instead of the next flash attn
                    # kernel, to minimize bubble when comm takes longer than attn.
                    _kv_comm(i + 1)

                    block_softmax_lse[i % 2] = (
                        block_softmax_lse[i % 2].transpose(0, 1).unsqueeze(-1).contiguous().float()
                    )  # (H, T) -> (T, H, 1)
                    assert block_out[i % 2].shape[:-1] == block_softmax_lse[i % 2].shape[:-1]

                    # Output and log sum exp correction.
                    # Ideally overlap this with the next flash attn kernel,
                    # since attn uses Tensor Core and rescale is element-wise, memory-bound and uses CUDA cores.
                    # (NOTE that this is the same as ping-pong scheduling idea in FA3)
                    # TODO However sometimes while the GPU has scheduled the next kernel,
                    # it's reluctant to launch it in overlap. Some potential causes:
                    # 1. need lower-level CUDA scheduling 2. further benchmark against Megatron-LM
                    # 3. register spilling by FA kernel.
                    if i == 0:
                        out = block_out[0]
                        softmax_lse = block_softmax_lse[0]
                    elif i <= local_sp_rank:
                        out, softmax_lse = _rescale_out_lse(
                            out, block_out[i % 2], softmax_lse, block_softmax_lse[i % 2]
                        )
                    else:
                        out[half_idx_back], softmax_lse[half_idx_back] = _rescale_out_lse(
                            out[half_idx_back], block_out[i % 2], softmax_lse[half_idx_back], block_softmax_lse[i % 2]
                        )

            torch.cuda.current_stream().wait_stream(sp_stream)
            return out, softmax_lse

        # Forward for inter-node (the outer ring in 2D ring)
        def _other_ring_forward(ring_num_idx, out, softmax_lse):
            # Loop through the inner ring after receiving
            # all new KVs from another ring
            for i in range(local_sp_size):
                with torch.cuda.stream(sp_streams[i % 2]):
                    # Send & recv KV
                    if i > 0:
                        local_kv_comms[(i + 1) % 2].wait()

                    # Prefetch
                    if i == 0:
                        _kv_comm(i)

                    if ring_num_idx > inter_ring_rank:
                        kv_block = kv_buffers[i % 2]
                        (
                            block_out[i % 2],
                            block_softmax_lse[i % 2],
                            rng_states[i + local_sp_size * ring_num_idx],
                        ) = _forward(q1, kv_block[0], kv_block[1], causal=False)
                        RingAttention.ATTN_DONE.record()

                        _kv_comm(i + 1)
                        block_softmax_lse[i % 2] = (
                            block_softmax_lse[i % 2].transpose(0, 1).unsqueeze(-1).contiguous().float()
                        )
                        out[half_idx_back], softmax_lse[half_idx_back] = _rescale_out_lse(
                            out[half_idx_back], block_out[i % 2], softmax_lse[half_idx_back], block_softmax_lse[i % 2]
                        )
                    else:
                        kv_block = kv_buffers[i % 2][:, half_idx_front]
                        (
                            block_out[i % 2],
                            block_softmax_lse[i % 2],
                            rng_states[i + local_sp_size * ring_num_idx],
                        ) = _forward(q, kv_block[0], kv_block[1], causal=False)
                        RingAttention.ATTN_DONE.record()

                        _kv_comm(i + 1)
                        block_softmax_lse[i % 2] = (
                            block_softmax_lse[i % 2].transpose(0, 1).unsqueeze(-1).contiguous().float()
                        )
                        out, softmax_lse = _rescale_out_lse(
                            out, block_out[i % 2], softmax_lse, block_softmax_lse[i % 2]
                        )

            torch.cuda.current_stream().wait_stream(sp_stream)
            return out, softmax_lse

        # Send and recv KV between rings at once to maximize NIC util.
        inter_ring_kv = None
        for ring_num_idx in range(num_rings):
            if ring_num_idx > 0:
                inter_ring_comm.wait()
                # Reset indices
                kv_buffers[0] = inter_ring_kv

            if ring_num_idx < num_rings - 1:
                if ring_num_idx == 0:
                    to_send = kv_buffers[0]
                else:
                    # The last received KV
                    to_send = kv_buffers[(local_sp_size - 1) % 2]
                inter_ring_kv = inter_ring_comm.send_recv(to_send)

            if ring_num_idx == 0:
                out, softmax_lse = _local_ring_forward()
            else:
                out, softmax_lse = _other_ring_forward(ring_num_idx, out, softmax_lse)

        out = out.to(q.dtype)
        if not is_packed:
            out = out.view(b, sq, h, d)
            q, k, v = [x.view(b, sq, *x.shape[-2:]) for x in (q, k, v)]  # (T, H, D) -> (B, Sq, H, D)
        softmax_lse = softmax_lse.squeeze(-1)

        ctx.sp_group = sp_group
        ctx.max_seqlen_q = ctx.max_seqlen_kv = max_seqlen
        misc_kwargs["deterministic"] = deterministic
        del misc_kwargs["return_softmax"]
        ctx.misc_kwargs = misc_kwargs
        ctx.is_packed = is_packed

        ctx.kv_group = inner_ring_group
        ctx.inter_kv_group = inter_ring_group

        ctx.save_for_backward(
            q,
            k,
            v,
            out,
            softmax_lse.transpose(0, 1).contiguous(),  # (T, H) -> (H, T)
            cu_seqlens_q,
            cu_seqlens_kv,
            half_idx_front,
            half_idx_back,
            *rng_states,
        )

        if return_softmax:
            return out, softmax_lse
        return out, None

    def backward(ctx, dout, _):
        """
        During backward, we accumulate q grads on each rank locally, but iterate kv and their grads
        over all ranks for accumulation. We avoid using two streams due to backward using doubled
        buffers and more comm cost.
        """
        (q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_kv, half_idx_front, half_idx_back) = ctx.saved_tensors[:9]
        rng_states = ctx.saved_tensors[9:]

        is_packed = ctx.is_packed
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_kv = ctx.max_seqlen_kv
        cu_seqlens_half = cu_seqlens_q // 2
        max_seqlen_half = max_seqlen_q // 2
        misc_kwargs = ctx.misc_kwargs
        del misc_kwargs["block_table"]

        assert (
            out.shape == dout.shape == q.shape
        ), f"out {out.shape} and dout {dout.shape} should have the same shape ({q.shape})."

        if is_packed:
            t, h, d = q.shape
        else:
            b, sq, h, d = q.shape
            t = b * sq
        q, k, v, out, dout = [x.view(t, *x.shape[-2:]) for x in (q, k, v, out, dout)]

        # Sequence parallel args
        sp_group = ctx.sp_group
        local_kv_group = ctx.kv_group
        inter_kv_group = ctx.inter_kv_group

        local_sp_rank = dist.get_rank(sp_group)
        sp_size = dist.get_world_size(sp_group)

        # NOTE: Using separate streams (PG) for concurrent kv and dkv comm may
        # cause NCCL "software caused connection abort" here...
        local_kv_comm = RingComm(local_kv_group)
        local_dkv_comm = RingComm(local_kv_group)
        inter_kv_comm = RingComm(inter_kv_group)
        inter_dkv_comm = RingComm(inter_kv_group)
        local_sp_size = dist.get_world_size(local_kv_group)
        local_sp_rank = dist.get_rank(local_kv_group)

        if dist.get_world_size(inter_kv_group) != sp_size:
            num_rings = dist.get_world_size(inter_kv_group)
            inter_ring_rank = dist.get_rank(inter_kv_group)
        else:
            num_rings = 1
            inter_ring_rank = 0

        if local_sp_rank != sp_size - 1:
            softmax_lse1 = softmax_lse[:, half_idx_back]
        dout = dout.contiguous()

        # Double comm buffers for sending and receiving kv
        kv_buffers = [torch.stack((k, v))]  # (2, T, H, D)
        kv_buffers.append(torch.empty_like(kv_buffers[0]))

        dq = None  # (T, H, D)
        # Intermediate outputs
        dq_block = torch.empty_like(q)  # (T, H, D)
        dk_block = torch.empty_like(k)  # (T, H, D)
        dv_block = torch.empty_like(v)  # (T, H, D)
        dkv_buffers = [torch.empty_like(kv, dtype=torch.float32) for kv in kv_buffers]  # (T, H, D)
        del k, v

        # Helper to pass args to FA
        def _backward(dout, q, k, v, out, softmax_lse, dq, dk, dv, rng_state, causal):
            _flash_attn_backward(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                cu_seqlens_q if dq.shape[0] == t else cu_seqlens_half,
                cu_seqlens_kv if dk.shape[0] == t else cu_seqlens_half,
                max_seqlen_q if dq.shape[0] == t else max_seqlen_half,
                max_seqlen_kv if dk.shape[0] == t else max_seqlen_half,
                causal=causal,
                rng_state=rng_state,
                **misc_kwargs,
            )

        # Backward within a node
        def _local_ring_backward():
            for i in range(local_sp_size):
                if i > 0:
                    local_kv_comm.wait()

                if i < local_sp_size - 1:
                    # Send kv to next rank for backward
                    local_kv_comm.send_recv(kv_buffers[i % 2], kv_buffers[(i + 1) % 2])

                if i == 0:
                    # Backward with local kv
                    k_, v_ = kv_buffers[i % 2]
                    q_, dout_, out_ = q, dout, out
                    dq_, dk_, dv_ = dq_block, dk_block, dv_block
                    _backward(dout_, q_, k_, v_, out_, softmax_lse, dq_, dk_, dv_, rng_states[i], causal=True)

                elif i <= local_sp_rank:
                    # Drop the second half of kv
                    # (T, H, D) -> (T // 2, H, D)
                    k_, v_ = [x[half_idx_front] for x in kv_buffers[i % 2]]
                    dk_, dv_ = [x[: t // 2] for x in (dk_block, dv_block)]
                    dq_, q_, out_, dout_ = (dq_block, q, out, dout)
                    _backward(dout_, q_, k_, v_, out_, softmax_lse, dq_, dk_, dv_, rng_states[i], causal=False)

                else:
                    # Drop the first half of q
                    k_, v_ = kv_buffers[i % 2]
                    dk_, dv_ = dk_block, dv_block
                    q_, out_, dout_ = [x[half_idx_back] for x in (q, out, dout)]
                    dq_ = dq_block[: t // 2]
                    _backward(dout_, q_, k_, v_, out_, softmax_lse1, dq_, dk_, dv_, rng_states[i], causal=False)

                # Accumulate grads
                if i == 0:
                    dq = dq_block.float()
                    dkv_buffers[i % 2][0] = dk_block.float()
                    dkv_buffers[i % 2][1] = dv_block.float()
                else:
                    # Accumulate local dq
                    if i <= local_sp_rank:
                        dq += dq_  # (T, H, D)
                    else:
                        dq[half_idx_back] += dq_

                    # Wait for mobile kv grad accumulators
                    local_dkv_comm.wait()

                    if i <= local_sp_rank:
                        # q blocks "surrounded" by kv blocks
                        dkv_buffers[i % 2][0][half_idx_front] += dk_
                        dkv_buffers[i % 2][1][half_idx_front] += dv_
                    else:
                        # q blocks "surrounding" kv blocks
                        dkv_buffers[i % 2][0] += dk_
                        dkv_buffers[i % 2][1] += dv_
                local_dkv_comm.send_recv(send_tensor=dkv_buffers[i % 2], recv_tensor=dkv_buffers[(i + 1) % 2])

            local_dkv_comm.wait()
            dkv_recv = dkv_buffers[local_sp_size % 2]
            dkv_send = dkv_buffers[(local_sp_size - 1) % 2]
            return dq, dkv_recv, dkv_send

        # Backward for inter-node (the outer ring in 2D ring)
        def _other_ring_backward(ring_num_idx, dq):
            if ring_num_idx > inter_ring_rank:
                # Indexing is expensive
                q_, out_, dout_ = [x[half_idx_back] for x in (q, out, dout)]
            else:
                q_, out_, dout_ = (q, out, dout)

            for i in range(local_sp_size):
                if i > 0:
                    local_kv_comm.wait()

                if i < local_sp_size - 1:
                    local_kv_comm.send_recv(kv_buffers[i % 2], kv_buffers[(i + 1) % 2])

                rng_state = rng_states[i + local_sp_size * ring_num_idx]
                if ring_num_idx > inter_ring_rank:
                    k_, v_ = kv_buffers[i % 2]
                    dk_, dv_ = dk_block, dv_block
                    dq_ = dq_block[: t // 2]
                    _backward(dout_, q_, k_, v_, out_, softmax_lse1, dq_, dk_, dv_, rng_state, causal=False)

                    dq[half_idx_back] += dq_
                    if i > 0:
                        local_dkv_comm.wait()
                    else:
                        inter_dkv_comm.wait()

                    dkv_buffers[i % 2][0] += dk_
                    dkv_buffers[i % 2][1] += dv_
                else:
                    k_, v_ = [x[half_idx_front] for x in kv_buffers[i % 2]]
                    dk_, dv_ = [x[: t // 2] for x in (dk_block, dv_block)]
                    dq_ = dq_block
                    _backward(dout_, q_, k_, v_, out_, softmax_lse, dq_, dk_, dv_, rng_state, causal=False)

                    dq += dq_
                    if i > 0:
                        local_dkv_comm.wait()
                    else:
                        inter_dkv_comm.wait()

                    dkv_buffers[i % 2][0][half_idx_front] += dk_
                    dkv_buffers[i % 2][1][half_idx_front] += dv_

                local_dkv_comm.send_recv(send_tensor=dkv_buffers[i % 2], recv_tensor=dkv_buffers[(i + 1) % 2])

            local_dkv_comm.wait()
            dkv_recv = dkv_buffers[local_sp_size % 2]
            dkv_send = dkv_buffers[(local_sp_size - 1) % 2]
            return dq, dkv_recv, dkv_send

        inter_ring_kv = None
        for ring_num_idx in range(num_rings):
            if ring_num_idx > 0:
                inter_kv_comm.wait()
                kv_buffers[0] = inter_ring_kv

            if ring_num_idx < num_rings - 1:
                # Re-allocate a buffer in each inter-ring step
                inter_ring_kv = inter_kv_comm.send_recv(kv_buffers[0])

            if ring_num_idx == 0:
                dq, dkv_recv, dkv_send = _local_ring_backward()
            else:
                dq, dkv_recv, dkv_send = _other_ring_backward(ring_num_idx, dq)

            if num_rings > 1:
                # Reuse the local buffers
                inter_dkv_comm.send_recv(send_tensor=dkv_recv, recv_tensor=dkv_send)
                # Reset indices
                dkv_buffers[0] = dkv_send
                dkv_buffers[1] = dkv_recv
                if ring_num_idx == num_rings - 1:
                    inter_dkv_comm.wait()
                    dkv_recv = dkv_buffers[0]

        dq, dk, dv = [x.to(q.dtype) for x in (dq, *dkv_recv)]
        if not is_packed:
            dq, dk, dv = [x.view(b, sq, *x.shape[-2:]) for x in (dq, dk, dv)]

        return (dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None)

    @staticmethod
    def prepare_varlen_batch(
        padding_mask: torch.Tensor,
        sp_group: dist.ProcessGroup,
        inputs_embeds: torch.Tensor = None,
        position_ids: Optional[torch.Tensor] = None,
        is_label: bool = False,
        is_batched_seq: bool = True,
    ):
        # TODO: support setting a batch dim (fix packing length) for packed mode, so that
        # DP can be used (needs to modify dataloader too)
        """
        Preprocess a batch of padded sequence by splitting input sequence by sp_size
        seq-wise and packing them into one sequence. Updates the mask info accordingly.
        Args:
            padding_mask (torch.Tensor): Contains the mask [B, Sq], where True means the token is NOT masked.
            sp_group (dist.ProcessGroup): Process group for sequence parallelism
            inputs_embeds (torch.Tensor): Input embeddings. Shape should be [B, Sq, ...]
            position_ids (Optional[torch.Tensor], optional): Position ids of shape [Sq] or [1, Sq]. Defaults to None.
            is_label (bool, optional): Whether inputs_embeds is instead a label tensor. If True, mask out the first
                token of each sequence.
            is_batched_seq (bool, optional): If True, then the input is a batch of (potentially padded) sequences
                of shape [B, Sq, ...]; else a packed sequence of shape [T, ...].

        Returns:
            inputs_embeds (torch.Tensor):
                Packed input embeddings of shape [B, Sq // sp_size, ...] if is_batched_seq, else [T, ...].
            mask_info (Dict[str, Any]):
                A dictionary containing mask info.
            position_ids (torch.Tensor):
                Packed position ids of shape [..., Sq // sp_size].

        """
        _load_varlen_helpers()
        sp_size = dist.get_world_size(group=sp_group)
        sp_rank = dist.get_rank(group=sp_group)
        mask_info = {}
        mask_info["max_seqlen"], mask_info["cu_seqlens"] = get_pad_info(padding_mask, return_indices=False)

        # Unpad, split seq-wise, then pad to (B, max_seqlen // sp_size)
        # (B, Sq) -> (B, max_seqlen // sp_size)
        padding_mask = padding_mask[:, : mask_info["max_seqlen"]]
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds[:, : mask_info["max_seqlen"]]
            inputs_embeds = split_varlen_zigzag(
                inputs_embeds,
                mask_info["cu_seqlens"],
                sp_group,
                mask_info["max_seqlen"],
                is_batched_seq=is_batched_seq,
                is_label=is_label,
            )
        # Split mask to get local nonzero seq positions
        padding_mask = split_varlen_zigzag(
            padding_mask, mask_info["cu_seqlens"], sp_group, mask_info["max_seqlen"], is_batched_seq=is_batched_seq
        )

        if position_ids is not None:
            indices = torch.tensor([sp_rank, 2 * sp_size - sp_rank - 1], device=inputs_embeds.device)
            position_ids = (
                position_ids[..., : mask_info["max_seqlen"]]  # unpad
                .view(-1, sp_size * 2, mask_info["max_seqlen"] // (sp_size * 2))
                .index_select(-2, indices)
                .view(-1, mask_info["max_seqlen"] // sp_size)
            )

        mask_info["max_seqlen"] //= sp_size
        mask_info["valid_indices"] = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
        mask_info["cu_seqlens"] //= sp_size
        mask_info["attention_mask_type"] = AttnMaskType.PADDED_CAUSAL
        return inputs_embeds, mask_info, position_ids

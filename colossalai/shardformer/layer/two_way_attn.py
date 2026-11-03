r"""Attention over a packed pair of autoregressive and diffusion sequences.

A world model such as Cosmos 3 runs one transformer layer over two kinds of tokens that share attention but not
visibility rules. The autoregressive (AR) part is causal; the diffusion (DM) part is bidirectional inside itself and
also sees the AR part; the AR part never sees the DM part. That is an asymmetric block mask, which is the one thing
a standard causal or full attention kernel cannot express.

The NVIDIA implementation splits the computation into two variable-length calls instead of materialising the block
mask:

1. the AR queries attend to the AR keys causally, delimited by ``cu_seqlens_ar``;
2. the DM queries attend to the per-sample concatenation ``[AR_i; DM_i]``, delimited by ``cu_seqlens_dm`` and
   ``cu_seqlens_kv``. The per-sample segments are the block diagonal of the mask, so a full attention kernel with
   variable-length boundaries is enough.

This module exposes that split as a thin orchestration layer. It does not change :class:`ColoAttention`: the naive
implementation is a plain PyTorch reference used to validate a fused implementation and to start SFT, and the NPU
implementation issues the two calls through the regular attention entry.

The packed layout is layout A of the design note: a global AR block followed by a global DM block. The first call
therefore needs no copy at all; the second call interleaves the K/V of the two blocks into the per-sample
``[AR_i; DM_i]`` order with a gather index that the packing step can build once per batch and reuse for every layer.
"""

import operator
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from .attn import AttnMaskType, ColoAttention
from .segmented_attn import SegmentMaskSpec, SegmentRule, naive_segment_attention

__all__ = [
    "TwoWayPackedMeta",
    "build_two_way_kv_gather_index",
    "validate_two_way_varlen_limits",
    "naive_two_way_flat_attention",
    "npu_two_way_flat_attention",
    "two_way_flat_attention",
]

# Limits of the Ascend varlen kernel. They are checked before a batch reaches the kernel, because once the kernel
# rejects a batch the error points at the operator rather than at the collate that produced the batch.
_MAX_SAMPLES = 2048
_MAX_PACKED_TOKENS = 1 << 20
_MAX_HEAD_DIM = 768
_MAX_QUERY_HEADS = 256


def _prefix_sums(lengths: Sequence[int]) -> List[int]:
    """Return ``[0, l0, l0 + l1, ...]`` for a sequence of lengths."""
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + int(length))
    return cumulative


def _decode_cu_seqlens(value, argument_name: str) -> List[int]:
    """Turn a cumulative-length prefix sum into per-sample lengths, allowing zero-length samples."""
    lengths = torch.as_tensor(value)
    if lengths.dim() != 1 or lengths.numel() == 0:
        raise ValueError(
            f"{argument_name} must be a 1D prefix sum [0, l0, l0 + l1, ...], got shape {tuple(lengths.shape)}."
        )
    if lengths.is_floating_point() or lengths.dtype is torch.bool:
        raise TypeError(
            f"{argument_name} must contain integers, got dtype {lengths.dtype}. Cast the lengths with "
            f".to(torch.int32) before building the metadata."
        )
    values = [int(length) for length in lengths.tolist()]
    if values[0] != 0:
        raise ValueError(f"{argument_name} must start at 0, got {values}.")
    if any(later < earlier for earlier, later in zip(values, values[1:])):
        raise ValueError(
            f"{argument_name} must be non-decreasing, got {values}. A shorter entry describes a zero-length sample, "
            f"which this layer drops from the corresponding kernel call."
        )
    return [later - earlier for earlier, later in zip(values, values[1:])]


@dataclass(frozen=True)
class TwoWayPackedMeta:
    """Layout of a packed two-way batch, in host memory.

    The canonical layout is a global AR block followed by a global DM block: sample ``i`` owns ``ar_lens[i]``
    contiguous tokens of the AR block and ``dm_lens[i]`` contiguous tokens of the DM block.

    A length may be zero. A pure-text sample has no DM tokens, a pure-generation sample has no AR tokens, and a
    sample with two zero lengths contributes nothing. Zero-length samples are dropped from the kernel calls, because
    ``cu_seqlens`` cannot describe an empty sequence, but their position in the batch is preserved so the packed
    tensors stay aligned.

    Args:
        ar_lens (Sequence[int]): Number of AR tokens of every sample, in batch order.
        dm_lens (Sequence[int]): Number of DM tokens of every sample, in batch order.

    Raises:
        ValueError: If the two lists have different lengths or if a length is negative.
        TypeError: If a length is not an integer.

    Example:
        .. code-block:: python

            from colossalai.shardformer.layer import TwoWayPackedMeta

            meta = TwoWayPackedMeta.from_cu_seqlens([0, 3, 8], [0, 5, 5])
            assert meta.ar_lens == (3, 5)
            assert meta.dm_lens == (5, 0)
            assert meta.cu_seqlens_kv == [0, 8, 13]
    """

    ar_lens: Tuple[int, ...]
    dm_lens: Tuple[int, ...]

    def __post_init__(self) -> None:
        try:
            # ``operator.index`` rejects 3.5 instead of truncating it to 3, while still accepting python ints, numpy
            # integers and zero-dimensional integer tensors.
            ar_lens = tuple(operator.index(length) for length in self.ar_lens)
            dm_lens = tuple(operator.index(length) for length in self.dm_lens)
        except TypeError as error:
            raise TypeError(
                f"ar_lens and dm_lens must be sequences of integers, got {self.ar_lens!r} and {self.dm_lens!r}."
            ) from error
        object.__setattr__(self, "ar_lens", ar_lens)
        object.__setattr__(self, "dm_lens", dm_lens)

        if len(ar_lens) != len(dm_lens):
            raise ValueError(
                f"ar_lens and dm_lens must describe the same number of samples, got {len(ar_lens)} and "
                f"{len(dm_lens)}."
            )
        if any(length < 0 for length in ar_lens) or any(length < 0 for length in dm_lens):
            raise ValueError(f"Every token count must be non-negative, got ar_lens={ar_lens} and dm_lens={dm_lens}.")

    @classmethod
    def from_cu_seqlens(
        cls,
        cu_seqlens_ar,
        cu_seqlens_dm,
        cu_seqlens_kv=None,
    ) -> "TwoWayPackedMeta":
        """Build the metadata from the cumulative lengths produced by the packing step.

        Args:
            cu_seqlens_ar: Prefix sum ``[0, ar_0, ar_0 + ar_1, ...]`` of the AR tokens.
            cu_seqlens_dm: Prefix sum ``[0, dm_0, dm_0 + dm_1, ...]`` of the DM tokens.
            cu_seqlens_kv (optional): Prefix sum of ``ar_i + dm_i``, the per-sample K/V length of the DM call. When it
                is given it is checked against the lengths of the two other arguments; when it is None it is derived.
                Defaults to None.

        Returns:
            TwoWayPackedMeta: The packed layout.

        Raises:
            ValueError: If a prefix sum is malformed or if ``cu_seqlens_kv`` does not describe ``ar_i + dm_i``.
            TypeError: If a prefix sum is not integral.
        """
        meta = cls(
            _decode_cu_seqlens(cu_seqlens_ar, "cu_seqlens_ar"),
            _decode_cu_seqlens(cu_seqlens_dm, "cu_seqlens_dm"),
        )
        if cu_seqlens_kv is not None:
            provided = _decode_cu_seqlens(cu_seqlens_kv, "cu_seqlens_kv")
            if len(provided) != meta.num_samples:
                raise ValueError(
                    f"cu_seqlens_kv describes {len(provided)} samples, but ar_lens and dm_lens describe "
                    f"{meta.num_samples}. Every prefix sum must come from the same collate."
                )
            expected = [ar_len + dm_len for ar_len, dm_len in zip(meta.ar_lens, meta.dm_lens)]
            if provided != expected:
                raise ValueError(
                    f"cu_seqlens_kv must describe ar_i + dm_i per sample, i.e. {expected}, got {provided}."
                )
        return meta

    @property
    def num_samples(self) -> int:
        """Number of samples in the batch."""
        return len(self.ar_lens)

    @property
    def total_ar_tokens(self) -> int:
        """Number of tokens in the global AR block."""
        return sum(self.ar_lens)

    @property
    def total_dm_tokens(self) -> int:
        """Number of tokens in the global DM block."""
        return sum(self.dm_lens)

    @property
    def total_kv_tokens(self) -> int:
        """Number of tokens in the global ``[AR; DM]`` sequence, before per-sample interleaving."""
        return self.total_ar_tokens + self.total_dm_tokens

    @property
    def max_ar(self) -> int:
        """Longest AR sequence of the batch, or 0 for an AR-less batch."""
        return max(self.ar_lens, default=0)

    @property
    def max_dm(self) -> int:
        """Longest DM sequence of the batch, or 0 for a DM-less batch."""
        return max(self.dm_lens, default=0)

    @property
    def max_kv(self) -> int:
        """Longest per-sample ``[AR_i; DM_i]`` sequence of the DM call."""
        return max((ar_len + dm_len for ar_len, dm_len in zip(self.ar_lens, self.dm_lens)), default=0)

    @property
    def cu_seqlens_ar(self) -> List[int]:
        """Prefix sum of the AR lengths, including zero-length samples."""
        return _prefix_sums(self.ar_lens)

    @property
    def cu_seqlens_dm(self) -> List[int]:
        """Prefix sum of the DM lengths, including zero-length samples."""
        return _prefix_sums(self.dm_lens)

    @property
    def cu_seqlens_kv(self) -> List[int]:
        """Prefix sum of the per-sample ``ar_i + dm_i`` lengths, including zero-length samples."""
        return _prefix_sums([ar_len + dm_len for ar_len, dm_len in zip(self.ar_lens, self.dm_lens)])

    def __repr__(self) -> str:
        return f"TwoWayPackedMeta(ar_lens={self.ar_lens}, dm_lens={self.dm_lens})"


def build_two_way_kv_gather_index(meta: TwoWayPackedMeta, device: Optional[torch.device] = None) -> torch.Tensor:
    """Index that interleaves the global AR and DM blocks into the per-sample ``[AR_i; DM_i]`` order.

    The source of the gather is ``torch.cat([k_ar, k_dm], dim=0)``. Applying the returned index to both K and V gives
    the K/V tensor of the second call. Samples without DM tokens are not part of that call and are omitted, which
    keeps the number of segments equal to ``cu_seqlens_dm``.

    The packing step can build this index once per batch and pass it to every layer, which is what makes the copy
    per layer a gather rather than a rebuild of the index as well.

    Args:
        meta (TwoWayPackedMeta): The packed layout.
        device (Optional[torch.device], optional): Device of the returned index. Defaults to None, i.e. CPU.

    Returns:
        torch.Tensor: Long index of shape ``[sum(ar_i + dm_i for samples with dm_i > 0)]``.

    Example:
        .. code-block:: python

            import torch

            from colossalai.shardformer.layer import TwoWayPackedMeta, build_two_way_kv_gather_index

            meta = TwoWayPackedMeta([3, 2], [6, 4])
            index = build_two_way_kv_gather_index(meta)
            assert index.tolist() == [0, 1, 2, 5, 6, 7, 8, 9, 10, 3, 4, 11, 12, 13, 14]
    """
    indices: List[torch.Tensor] = []
    ar_offset = 0
    dm_offset = meta.total_ar_tokens
    for ar_len, dm_len in zip(meta.ar_lens, meta.dm_lens):
        if dm_len == 0:
            ar_offset += ar_len
            continue
        if ar_len > 0:
            indices.append(torch.arange(ar_offset, ar_offset + ar_len, device=device))
        indices.append(torch.arange(dm_offset, dm_offset + dm_len, device=device))
        ar_offset += ar_len
        dm_offset += dm_len

    if not indices:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.cat(indices)


def validate_two_way_varlen_limits(
    meta: TwoWayPackedMeta,
    num_heads: int,
    head_dim: int,
    num_kv_heads: Optional[int] = None,
) -> None:
    """Check the Ascend varlen limits at collate time, before a batch reaches the kernel.

    The kernel accepts at most 2048 sequences, 1M packed tokens, 256 query heads and a head dimension of 768, and a
    GQA ratio that divides evenly. Checking here turns an operator error into a message that names the offending
    field, and keeps a malformed batch out of the training step.

    Args:
        meta (TwoWayPackedMeta): The packed layout.
        num_heads (int): Number of query heads.
        head_dim (int): Head dimension.
        num_kv_heads (Optional[int], optional): Number of key/value heads. When given, it must divide ``num_heads``
            evenly. Defaults to None, i.e. multi-head attention.

    Raises:
        ValueError: If the batch exceeds a kernel limit, or if a sample contributes no token at all. A sample with
            no token cannot be represented by ``cu_seqlens`` and would silently disappear from the batch.

    Example:
        .. code-block:: python

            from colossalai.shardformer.layer import TwoWayPackedMeta, validate_two_way_varlen_limits

            validate_two_way_varlen_limits(TwoWayPackedMeta([3, 5], [5, 7]), num_heads=32, head_dim=128)
    """
    if meta.num_samples > _MAX_SAMPLES:
        raise ValueError(
            f"The varlen kernel accepts at most {_MAX_SAMPLES} sequences per batch, got {meta.num_samples}. Split "
            f"the batch into buckets before collating."
        )
    if meta.total_kv_tokens > _MAX_PACKED_TOKENS:
        raise ValueError(
            f"The varlen kernel accepts at most {_MAX_PACKED_TOKENS} packed tokens, got {meta.total_kv_tokens} "
            f"(ar={meta.total_ar_tokens}, dm={meta.total_dm_tokens}). Lower the token budget or split the batch."
        )
    if not 1 <= head_dim <= _MAX_HEAD_DIM:
        raise ValueError(f"The head dimension must be in [1, {_MAX_HEAD_DIM}], got {head_dim}.")
    if not 1 <= num_heads <= _MAX_QUERY_HEADS:
        raise ValueError(f"The number of query heads must be in [1, {_MAX_QUERY_HEADS}], got {num_heads}.")
    if num_kv_heads is not None and (num_kv_heads <= 0 or num_heads % num_kv_heads != 0):
        raise ValueError(
            f"The number of query heads ({num_heads}) must be a positive multiple of the key/value heads "
            f"({num_kv_heads})."
        )

    empty_samples = [
        index for index, (ar_len, dm_len) in enumerate(zip(meta.ar_lens, meta.dm_lens)) if ar_len + dm_len == 0
    ]
    if empty_samples:
        raise ValueError(
            f"Samples {empty_samples[:8]} contribute no token at all. A zero-length sequence cannot be expressed by "
            f"cu_seqlens and would silently vanish from the batch; drop these samples before collating."
        )


def _check_packed_inputs(
    q_ar: torch.Tensor,
    k_ar: torch.Tensor,
    v_ar: torch.Tensor,
    q_dm: torch.Tensor,
    k_dm: torch.Tensor,
    v_dm: torch.Tensor,
    meta: TwoWayPackedMeta,
) -> None:
    """Validate the packed tensors against the metadata and against each other."""
    expected = {
        "q_ar": (q_ar, meta.total_ar_tokens),
        "k_ar": (k_ar, meta.total_ar_tokens),
        "v_ar": (v_ar, meta.total_ar_tokens),
        "q_dm": (q_dm, meta.total_dm_tokens),
        "k_dm": (k_dm, meta.total_dm_tokens),
        "v_dm": (v_dm, meta.total_dm_tokens),
    }
    for name, (tensor, tokens) in expected.items():
        if tensor.dim() != 3:
            raise ValueError(
                f"{name} must be a packed [tokens, heads, head_dim] tensor, got shape {tuple(tensor.shape)}."
            )
        if tensor.size(0) != tokens:
            raise ValueError(
                f"{name} holds {tensor.size(0)} tokens, but the metadata describes {tokens}. The packed tensors and "
                f"the metadata must come from the same collate."
            )

    if k_ar.shape != v_ar.shape:
        raise ValueError(f"k_ar and v_ar must have the same shape, got {tuple(k_ar.shape)} and {tuple(v_ar.shape)}.")
    if k_dm.shape != v_dm.shape:
        raise ValueError(f"k_dm and v_dm must have the same shape, got {tuple(k_dm.shape)} and {tuple(v_dm.shape)}.")
    if q_ar.shape[1:] != q_dm.shape[1:]:
        raise ValueError(
            f"The AR and DM towers must agree on heads and head_dim, got {tuple(q_ar.shape[1:])} and "
            f"{tuple(q_dm.shape[1:])}."
        )
    if k_ar.shape[1:] != k_dm.shape[1:]:
        raise ValueError(
            f"The AR and DM towers must agree on key/value heads and head_dim, got {tuple(k_ar.shape[1:])} and "
            f"{tuple(k_dm.shape[1:])}."
        )
    num_q_heads, head_dim = q_ar.size(1), q_ar.size(2)
    num_kv_heads = k_ar.size(1)
    if num_q_heads == 0 or num_kv_heads == 0:
        raise ValueError(f"q and k must have at least one head, got {num_q_heads} query and {num_kv_heads} key/value.")
    if head_dim != k_ar.size(2):
        raise ValueError(f"q and k must agree on head_dim, got {head_dim} and {k_ar.size(2)}.")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"Grouped-query attention requires the number of query heads ({num_q_heads}) to be a multiple of the "
            f"number of key/value heads ({num_kv_heads})."
        )


def _two_way_segment_spec(ar_len: int, dm_len: int) -> SegmentMaskSpec:
    """The reference mask of one sample: AR causal, DM full, AR never sees DM."""
    if ar_len > 0 and dm_len > 0:
        return SegmentMaskSpec(
            [ar_len, dm_len],
            [[SegmentRule.CAUSAL, SegmentRule.NONE], [SegmentRule.FULL, SegmentRule.FULL]],
        )
    if ar_len > 0:
        return SegmentMaskSpec([ar_len], [[SegmentRule.CAUSAL]])
    return SegmentMaskSpec([dm_len], [[SegmentRule.FULL]])


def naive_two_way_flat_attention(
    q_ar: torch.Tensor,
    k_ar: torch.Tensor,
    v_ar: torch.Tensor,
    q_dm: torch.Tensor,
    k_dm: torch.Tensor,
    v_dm: torch.Tensor,
    meta: TwoWayPackedMeta,
    *,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
    kv_gather_index: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference two-way attention, one plain PyTorch attention call per sample.

    This is the independent implementation a fused path is checked against, and the fallback that lets SFT start
    before the fused kernel is ready. It materialises the ``[S_i, S_i]`` scores of each sample, so it is exact and
    device agnostic but slow; it is not meant for long runs. ``float64`` inputs are kept in ``float64``.

    Args:
        q_ar (torch.Tensor): AR queries of shape ``[total_ar_tokens, heads, head_dim]``.
        k_ar (torch.Tensor): AR keys of shape ``[total_ar_tokens, kv_heads, head_dim]``; ``kv_heads`` may be smaller
            than the query ``heads``, as in grouped-query attention.
        v_ar (torch.Tensor): AR values, same shape as ``k_ar``.
        q_dm (torch.Tensor): DM queries of shape ``[total_dm_tokens, heads, head_dim]``.
        k_dm (torch.Tensor): DM keys, same shape as ``k_ar``.
        v_dm (torch.Tensor): DM values, same shape as ``k_dm``.
        meta (TwoWayPackedMeta): Packed layout that describes where each sample starts and ends.
        scale (Optional[float], optional): Multiplier applied to the scores before the softmax. Defaults to
            ``head_dim**-0.5``.
        dropout_p (float, optional): Dropout on the attention probabilities. Keep the default of 0.0 for a reference
            run used in a numerical comparison. Defaults to 0.0.
        kv_gather_index (Optional[torch.Tensor], optional): Accepted for signature parity with
            :func:`npu_two_way_flat_attention`. The reference reads the two K/V blocks directly and does not need the
            interleaved view. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: AR output of shape ``[total_ar_tokens, heads, head_dim]`` and DM output of
        shape ``[total_dm_tokens, heads, head_dim]``.

    Raises:
        ValueError: If the tensors do not match the metadata, or if the towers disagree on heads or ``head_dim``.

    Example:
        .. code-block:: python

            import torch

            from colossalai.shardformer.layer import TwoWayPackedMeta, naive_two_way_flat_attention

            meta = TwoWayPackedMeta([3], [5])
            q_ar, k_ar, v_ar = (torch.randn(3, 2, 16, dtype=torch.float64) for _ in range(3))
            q_dm, k_dm, v_dm = (torch.randn(5, 2, 16, dtype=torch.float64) for _ in range(3))
            out_ar, out_dm = naive_two_way_flat_attention(q_ar, k_ar, v_ar, q_dm, k_dm, v_dm, meta)
            assert out_ar.shape == (3, 2, 16) and out_dm.shape == (5, 2, 16)
    """
    _check_packed_inputs(q_ar, k_ar, v_ar, q_dm, k_dm, v_dm, meta)

    ar_outputs: List[torch.Tensor] = []
    dm_outputs: List[torch.Tensor] = []
    ar_offset = 0
    dm_offset = 0
    for ar_len, dm_len in zip(meta.ar_lens, meta.dm_lens):
        if ar_len + dm_len == 0:
            continue
        q = torch.cat((q_ar[ar_offset : ar_offset + ar_len], q_dm[dm_offset : dm_offset + dm_len]), dim=0)
        k = torch.cat((k_ar[ar_offset : ar_offset + ar_len], k_dm[dm_offset : dm_offset + dm_len]), dim=0)
        v = torch.cat((v_ar[ar_offset : ar_offset + ar_len], v_dm[dm_offset : dm_offset + dm_len]), dim=0)
        sample_output = naive_segment_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            _two_way_segment_spec(ar_len, dm_len),
            scale=scale,
            dropout_p=dropout_p,
        )[0].transpose(0, 1)
        ar_outputs.append(sample_output[:ar_len])
        dm_outputs.append(sample_output[ar_len:])
        ar_offset += ar_len
        dm_offset += dm_len

    out_ar = torch.cat(ar_outputs, dim=0) if ar_outputs else q_ar.new_empty(q_ar.shape)
    out_dm = torch.cat(dm_outputs, dim=0) if dm_outputs else q_dm.new_empty(q_dm.shape)
    return out_ar, out_dm


def npu_two_way_flat_attention(
    q_ar: torch.Tensor,
    k_ar: torch.Tensor,
    v_ar: torch.Tensor,
    q_dm: torch.Tensor,
    k_dm: torch.Tensor,
    v_dm: torch.Tensor,
    meta: TwoWayPackedMeta,
    *,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
    kv_gather_index: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Two-way attention as two variable-length kernel calls on an Ascend NPU.

    The first call is a causal packed call over the global AR block. The second is a full packed call whose queries
    are the DM block and whose keys/values are the per-sample ``[AR_i; DM_i]`` interleaving. The block diagonal of
    the DM mask is expressed by ``actual_seq_qlen``/``actual_seq_kvlen``, so no ``[S, S]`` block mask is built. The
    causal mask of the first call is owned and cached by the NPU backend, not rebuilt here.

    The interleaving costs one concatenation plus one selection per K/V tensor. Pass the index from the packing
    step so it is not rebuilt per layer, and measure its share of the step time; if it exceeds the design note's
    budget, replace the concatenation with a destination-index write or fold the interleaving into the projection's
    output.

    Args:
        q_ar (torch.Tensor): AR queries of shape ``[total_ar_tokens, heads, head_dim]``.
        k_ar (torch.Tensor): AR keys of shape ``[total_ar_tokens, kv_heads, head_dim]``; ``kv_heads`` may be smaller
            than the query ``heads``, as in grouped-query attention.
        v_ar (torch.Tensor): AR values, same shape as ``k_ar``.
        q_dm (torch.Tensor): DM queries of shape ``[total_dm_tokens, heads, head_dim]``.
        k_dm (torch.Tensor): DM keys, same shape as ``k_ar``.
        v_dm (torch.Tensor): DM values, same shape as ``k_dm``.
        meta (TwoWayPackedMeta): Packed layout that describes where each sample starts and ends.
        scale (Optional[float], optional): Multiplier applied to the scores before the softmax. Defaults to
            ``head_dim**-0.5``.
        dropout_p (float, optional): Dropout probability passed to both calls. Defaults to 0.0.
        kv_gather_index (Optional[torch.Tensor], optional): Interleaving index returned by
            :func:`build_two_way_kv_gather_index`. Build it once per batch and reuse it for every layer; when it is
            None it is built here. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: AR output of shape ``[total_ar_tokens, heads, head_dim]`` and DM output of
        shape ``[total_dm_tokens, heads, head_dim]``.

    Raises:
        ValueError: If the tensors do not match the metadata, if the towers disagree on heads or ``head_dim``, if a
            gather index does not match the metadata, or if the attention entry cannot serve the packed causal call.

    Example:
        .. code-block:: python

            from colossalai.shardformer.layer import (
                TwoWayPackedMeta,
                build_two_way_kv_gather_index,
                npu_two_way_flat_attention,
            )

            meta = TwoWayPackedMeta([3], [5])
            gather_index = build_two_way_kv_gather_index(meta, device=q_dm.device)
            out_ar, out_dm = npu_two_way_flat_attention(
                q_ar, k_ar, v_ar, q_dm, k_dm, v_dm, meta, kv_gather_index=gather_index
            )
    """
    _check_packed_inputs(q_ar, k_ar, v_ar, q_dm, k_dm, v_dm, meta)

    ar_lengths = [ar_len for ar_len in meta.ar_lens if ar_len > 0]
    dm_lengths = [dm_len for dm_len in meta.dm_lens if dm_len > 0]
    kv_lengths = [ar_len + dm_len for ar_len, dm_len in zip(meta.ar_lens, meta.dm_lens) if dm_len > 0]

    if ar_lengths:
        cu_seqlens_ar = torch.tensor(_prefix_sums(ar_lengths), dtype=torch.int32, device=q_ar.device)
        out_ar = ColoAttention.attention(
            q_ar,
            k_ar,
            v_ar,
            attention_mask=None,
            attention_mask_type=AttnMaskType.CAUSAL,
            cu_seqlens_q=cu_seqlens_ar,
            cu_seqlens_kv=cu_seqlens_ar,
            max_seqlen_q=meta.max_ar,
            max_seqlen_kv=meta.max_ar,
            dropout_p=dropout_p,
            scale=scale,
            qkv_layout="TND",
        )
    else:
        out_ar = q_ar.new_empty(q_ar.shape)

    if dm_lengths:
        cu_seqlens_dm = torch.tensor(_prefix_sums(dm_lengths), dtype=torch.int32, device=q_dm.device)
        cu_seqlens_kv = torch.tensor(_prefix_sums(kv_lengths), dtype=torch.int32, device=q_dm.device)
        if kv_gather_index is None:
            kv_gather_index = build_two_way_kv_gather_index(meta, device=q_dm.device)
        expected_kv_tokens = sum(kv_lengths)
        if kv_gather_index.numel() != expected_kv_tokens:
            raise ValueError(
                f"kv_gather_index selects {kv_gather_index.numel()} tokens, but the DM call needs "
                f"{expected_kv_tokens} (sum of ar_i + dm_i over samples with dm_i > 0). Rebuild it with "
                f"build_two_way_kv_gather_index."
            )
        interleaved_keys = torch.cat((k_ar, k_dm), dim=0).index_select(0, kv_gather_index)
        interleaved_values = torch.cat((v_ar, v_dm), dim=0).index_select(0, kv_gather_index)
        out_dm = ColoAttention.attention(
            q_dm,
            interleaved_keys,
            interleaved_values,
            attention_mask=None,
            attention_mask_type=AttnMaskType.CUSTOM,
            cu_seqlens_q=cu_seqlens_dm,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=meta.max_dm,
            max_seqlen_kv=max(kv_lengths),
            dropout_p=dropout_p,
            scale=scale,
            qkv_layout="TND",
        )
    else:
        out_dm = q_dm.new_empty(q_dm.shape)

    return out_ar, out_dm


def two_way_flat_attention(
    q_ar: torch.Tensor,
    k_ar: torch.Tensor,
    v_ar: torch.Tensor,
    q_dm: torch.Tensor,
    k_dm: torch.Tensor,
    v_dm: torch.Tensor,
    meta: TwoWayPackedMeta,
    *,
    backend: str = "naive",
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
    kv_gather_index: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run two-way flat attention with the selected implementation.

    The naive and the NPU implementations share this signature, so a model can switch between them with one flag:
    use ``naive`` to start SFT and to produce the reference, and ``npu`` once the fused path has passed the numerical
    gate. The model keeps the two towers' Q/K/V projections separate and only hands the packed tensors here.

    Args:
        q_ar (torch.Tensor): AR queries of shape ``[total_ar_tokens, heads, head_dim]``.
        k_ar (torch.Tensor): AR keys of shape ``[total_ar_tokens, kv_heads, head_dim]``; ``kv_heads`` may be smaller
            than the query ``heads``, as in grouped-query attention.
        v_ar (torch.Tensor): AR values, same shape as ``k_ar``.
        q_dm (torch.Tensor): DM queries of shape ``[total_dm_tokens, heads, head_dim]``.
        k_dm (torch.Tensor): DM keys, same shape as ``k_ar``.
        v_dm (torch.Tensor): DM values, same shape as ``k_dm``.
        meta (TwoWayPackedMeta): Packed layout that describes where each sample starts and ends.
        backend (str, optional): ``"naive"`` for the PyTorch reference or ``"npu"`` for the two-call Ascend path.
            Defaults to "naive".
        scale (Optional[float], optional): Multiplier applied to the scores before the softmax. Defaults to
            ``head_dim**-0.5``.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.
        kv_gather_index (Optional[torch.Tensor], optional): Interleaving index for the NPU path; ignored by the naive
            path. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: AR output of shape ``[total_ar_tokens, heads, head_dim]`` and DM output of
        shape ``[total_dm_tokens, heads, head_dim]``.

    Raises:
        ValueError: If ``backend`` is unknown, or if the inputs do not match the metadata.

    Example:
        .. code-block:: python

            from colossalai.shardformer.layer import TwoWayPackedMeta, two_way_flat_attention

            meta = TwoWayPackedMeta([3], [5])
            out_ar, out_dm = two_way_flat_attention(q_ar, k_ar, v_ar, q_dm, k_dm, v_dm, meta)
    """
    implementations = {
        "naive": naive_two_way_flat_attention,
        "npu": npu_two_way_flat_attention,
    }
    implementation = implementations.get(backend)
    if implementation is None:
        raise ValueError(
            f"Unsupported two-way attention backend {backend!r}; expected one of {sorted(implementations)}."
        )
    return implementation(
        q_ar,
        k_ar,
        v_ar,
        q_dm,
        k_dm,
        v_dm,
        meta,
        scale=scale,
        dropout_p=dropout_p,
        kv_gather_index=kv_gather_index,
    )

"""Attention over a packed sequence whose segments follow different masking rules.

Multimodal and world-model training pack several kinds of tokens into a single sequence and then need a different
visibility rule for every part: a causal block followed by a bidirectional block, with cross-part visibility that is
neither ``CAUSAL`` nor ``CUSTOM`` as ``AttnMaskType`` defines them. ``ColoAttention.prepare_attn_kwargs`` cannot build
such a mask, because it derives the mask from a padding mask plus one "is this causal" flag.

This module keeps the description explicit instead of hiding it in model code. :class:`SegmentMaskSpec` records the
segment lengths and the rule between every pair of segments, :func:`build_segment_mask` turns that into the dense
boolean mask the existing attention kernels accept, and :func:`naive_segment_attention` is a plain PyTorch
implementation of the same math. The naive implementation exists to be the independent reference a fused kernel is
checked against, so it is written without any device-specific code and runs in float64 as well as in bf16/fp16.
"""

from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

__all__ = ["SegmentRule", "SegmentMaskSpec", "build_segment_mask", "naive_segment_attention"]


class SegmentRule(Enum):
    """How query positions of one segment see key positions of another.

    Attributes:
        FULL: Every query sees every key of the block.
        CAUSAL: A query sees the keys of the block whose in-block index is not larger than its own, i.e. the block is
            masked with a lower-left triangle. For blocks that are not square this is a left-aligned triangle.
        NONE: A query sees no key of the block.
    """

    FULL = "full"
    CAUSAL = "causal"
    NONE = "none"


class SegmentMaskSpec:
    """Segments of a packed sequence and the visibility rule between every pair of segments.

    Args:
        segment_lengths (Sequence[int]): Length of each segment, in sequence order. Every length must be positive.
        rules (Sequence[Sequence[SegmentRule]]): Square matrix with one row and one column per segment, where
            ``rules[i][j]`` is the rule for queries of segment ``i`` against keys of segment ``j``.

    Raises:
        ValueError: If there is no segment, if a length is not positive, if ``rules`` is not square with one row and
            one column per segment, or if a segment would have no visible key at all.

    Example:
        A three-token causal prefix followed by a five-token bidirectional suffix, with the two parts seeing each
        other:

        .. code-block:: python

            from colossalai.shardformer.layer import SegmentMaskSpec, SegmentRule, build_segment_mask

            spec = SegmentMaskSpec(
                segment_lengths=[3, 5],
                rules=[
                    [SegmentRule.CAUSAL, SegmentRule.FULL],
                    [SegmentRule.FULL, SegmentRule.FULL],
                ],
            )
            mask = build_segment_mask(spec)  # [8, 8], True means "may attend"
    """

    def __init__(
        self,
        segment_lengths: Sequence[int],
        rules: Sequence[Sequence[SegmentRule]],
    ) -> None:
        lengths = [int(length) for length in segment_lengths]
        if len(lengths) == 0:
            raise ValueError("SegmentMaskSpec needs at least one segment.")
        if any(length <= 0 for length in lengths):
            raise ValueError(f"Every segment length must be positive, got segment_lengths={lengths}.")

        rule_matrix = [[SegmentRule(rule) for rule in row] for row in rules]
        widths = [len(row) for row in rule_matrix]
        if len(rule_matrix) != len(lengths) or any(width != len(lengths) for width in widths):
            raise ValueError(
                f"rules must be a square matrix with one row and one column per segment, i.e. "
                f"{len(lengths)}x{len(lengths)}; got {len(rule_matrix)} row(s) of width(s) {widths}."
            )
        for index in range(len(lengths)):
            if rule_matrix[index][index] is SegmentRule.NONE:
                raise ValueError(
                    f"Segment {index} of lengths={lengths} would have no visible key at all, which makes its "
                    f"attention output undefined; a token must be able to attend to itself, so the diagonal rule "
                    f"cannot be SegmentRule.NONE."
                )

        self._segment_lengths = lengths
        self._rules = rule_matrix

    @property
    def segment_lengths(self) -> List[int]:
        """Length of every segment, in sequence order."""
        return list(self._segment_lengths)

    @property
    def rules(self) -> List[List[SegmentRule]]:
        """The rule matrix, one row and one column per segment."""
        return [list(row) for row in self._rules]

    @property
    def total_length(self) -> int:
        """Number of tokens of the packed sequence."""
        return sum(self._segment_lengths)

    @property
    def offsets(self) -> List[int]:
        """Start index of every segment inside the packed sequence."""
        offsets = []
        running_total = 0
        for length in self._segment_lengths:
            offsets.append(running_total)
            running_total += length
        return offsets

    def __repr__(self) -> str:
        rules = [[rule.value for rule in row] for row in self._rules]
        return f"SegmentMaskSpec(segment_lengths={self._segment_lengths}, rules={rules})"


def build_segment_mask(spec: SegmentMaskSpec, device: Optional[torch.device] = None) -> torch.Tensor:
    """Build the dense visibility mask described by ``spec``.

    Args:
        spec (SegmentMaskSpec): The segments and the rule between every pair of them.
        device (Optional[torch.device], optional): Device of the returned mask. Defaults to None, i.e. CPU.

    Returns:
        torch.Tensor: Boolean mask of shape ``[total_length, total_length]`` where ``True`` means that a key may be
        attended by a query. It can be fed to :func:`naive_segment_attention` directly, and to
        ``ColoAttention.attention`` after ``invert_mask(mask.unsqueeze(0).unsqueeze(0).to(dtype))``.

    Example:
        .. code-block:: python

            from colossalai.shardformer.layer import SegmentMaskSpec, SegmentRule, build_segment_mask

            spec = SegmentMaskSpec([2, 2], [[SegmentRule.CAUSAL, SegmentRule.NONE], [SegmentRule.NONE, SegmentRule.FULL]])
            print(build_segment_mask(spec).int())
            # tensor([[1, 0, 0, 0],
            #         [1, 1, 0, 0],
            #         [0, 0, 1, 1],
            #         [0, 0, 1, 1]])
    """
    mask = torch.zeros(spec.total_length, spec.total_length, dtype=torch.bool, device=device)
    for query_index, query_offset in enumerate(spec.offsets):
        query_length = spec.segment_lengths[query_index]
        for key_index, key_offset in enumerate(spec.offsets):
            rule = spec.rules[query_index][key_index]
            if rule is SegmentRule.NONE:
                continue
            key_length = spec.segment_lengths[key_index]
            block = torch.ones(query_length, key_length, dtype=torch.bool, device=device)
            if rule is SegmentRule.CAUSAL:
                block.tril_()
            mask[query_offset : query_offset + query_length, key_offset : key_offset + key_length] = block
    return mask


def naive_segment_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[Union[torch.Tensor, SegmentMaskSpec]] = None,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
    return_scores: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Plain PyTorch attention under a boolean visibility mask.

    This is the reference implementation for segmented and block structured masks: it materialises the full
    ``[Sq, Skv]`` score matrix, so it is slow and memory hungry, but it is exact, device agnostic and runs in float64.

    Args:
        q (torch.Tensor): Query of shape ``[batch, heads, Sq, head_dim]``.
        k (torch.Tensor): Key of shape ``[batch, heads, Skv, head_dim]``.
        v (torch.Tensor): Value of shape ``[batch, heads, Skv, head_dim]``.
        attention_mask (Optional[Union[torch.Tensor, SegmentMaskSpec]], optional): Boolean mask where ``True`` means
            "may attend", of shape ``[Sq, Skv]``, ``[batch, Sq, Skv]``, ``[1, 1, Sq, Skv]`` or
            ``[batch, 1, Sq, Skv]``; a :class:`SegmentMaskSpec` is built on the device of ``q``. ``None`` means every
            key is visible. Defaults to None.
        scale (Optional[float], optional): Multiplier applied to the scores before the softmax. Defaults to
            ``head_dim**-0.5``.
        dropout_p (float, optional): Dropout probability on the attention probabilities. Dropout is only applied when
            it is positive, so a reference run used for comparison should keep the default of 0.0. Defaults to 0.0.
        return_scores (bool, optional): Also return the masked scores, i.e. what the softmax consumes. This is what
            makes "was every position that must be masked actually masked" checkable without looking inside a fused
            kernel. Defaults to False.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: The attention output of shape
        ``[batch, heads, Sq, head_dim]``, and additionally the masked scores of shape ``[batch, heads, Sq, Skv]`` when
        ``return_scores`` is True.

    Raises:
        ValueError: If ``q``, ``k`` and ``v`` do not describe a consistent attention, if the mask does not match the
            sequence lengths, or if a query row has no visible key at all, which would make its softmax undefined.
        TypeError: If ``attention_mask`` is a tensor that is not a boolean mask.

    Example:
        .. code-block:: python

            import torch

            from colossalai.shardformer.layer import SegmentMaskSpec, SegmentRule, naive_segment_attention

            spec = SegmentMaskSpec([3, 5], [[SegmentRule.CAUSAL, SegmentRule.FULL], [SegmentRule.FULL, SegmentRule.FULL]])
            q, k, v = (torch.randn(1, 2, 8, 16, dtype=torch.float64) for _ in range(3))
            out, scores = naive_segment_attention(q, k, v, spec, return_scores=True)
            assert torch.isinf(scores[0, 0, 0, 1])  # the first causal token cannot see the second one
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"q, k and v must be 4D tensors of shape [batch, heads, sequence, head_dim], got shapes "
            f"{tuple(q.shape)}, {tuple(k.shape)} and {tuple(v.shape)}."
        )
    if q.shape[:2] != k.shape[:2] or k.shape[:2] != v.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError(
            f"q, k and v must agree on batch, heads and head_dim, got shapes {tuple(q.shape)}, {tuple(k.shape)} "
            f"and {tuple(v.shape)}."
        )
    if k.shape[-1] != v.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError(
            f"k and v must have the same sequence length and head_dim, got {tuple(k.shape)} and {tuple(v.shape)}."
        )

    batch_size, _, s_q, head_dim = q.shape
    s_kv = k.shape[-2]

    if isinstance(attention_mask, SegmentMaskSpec):
        attention_mask = build_segment_mask(attention_mask, device=q.device)
    if attention_mask is not None:
        if attention_mask.dtype is not torch.bool:
            raise TypeError(
                f"attention_mask must be a boolean mask where True means 'may attend', got dtype "
                f"{attention_mask.dtype}. Build one with build_segment_mask, or cast with mask.bool() if it already "
                f"follows that convention."
            )
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        if (
            attention_mask.dim() != 4
            or tuple(attention_mask.shape[-2:]) != (s_q, s_kv)
            or attention_mask.shape[0] not in (1, batch_size)
        ):
            raise ValueError(
                f"attention_mask must be broadcastable to {(batch_size, 1, s_q, s_kv)} for q of shape "
                f"{tuple(q.shape)} and k of shape {tuple(k.shape)}, got {tuple(attention_mask.shape)}."
            )
        fully_masked_rows = torch.nonzero((~attention_mask).all(dim=-1), as_tuple=False)
        if fully_masked_rows.numel() > 0:
            raise ValueError(
                f"attention_mask leaves {fully_masked_rows.size(0)} query row(s) without a visible key, e.g. "
                f"{fully_masked_rows[:4].tolist()} as (batch, head, query) of the mask; their softmax would be all "
                f"-inf and produce NaN. Give every query at least one visible key, or drop those rows before calling."
            )

    if scale is None:
        scale = head_dim**-0.5

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if attention_mask is not None:
        scores = scores.masked_fill(~attention_mask, float("-inf"))
    # Accumulate half precision attention in float32, and keep the input dtype otherwise: a float64 call must stay
    # exact instead of losing the extra digits to a float32 round trip.
    softmax_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    probabilities = torch.softmax(scores, dim=-1, dtype=softmax_dtype).to(q.dtype)
    probabilities = F.dropout(probabilities, p=dropout_p, training=dropout_p > 0.0)
    output = torch.matmul(probabilities, v)

    if return_scores:
        return output, scores
    return output

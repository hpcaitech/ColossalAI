r"""Packing adapter for Cosmos-style MoT (mixture-of-transformers) training.

A Cosmos 3 packer hands the model one flat sequence per batch, ordered per sample as ``[AR_i | DM_i]``: each sample
contributes a causal autoregressive (understanding) split followed by a full diffusion (generation) split, and a
trailing pad segment may follow. The two-way attention kernel in
:mod:`colossalai.shardformer.layer.two_way_attn` instead takes a global AR block followed by a global DM block, plus
a :class:`~colossalai.shardformer.layer.two_way_attn.TwoWayPackedMeta`.

:class:`CosmosTwoWayLayout` is the seam between those two contracts. It is built from the packer's split
description, converts a packed tensor into the kernel's two blocks, and puts the kernel's two outputs back into pack
order. It lives here rather than in the attention layer so the kernel stays independent of any one model's packer.

The pad segment belongs to the packer, not to the kernel: the kernel cannot describe a segment that belongs to no
sample. Pass the real splits only (the Cosmos packer exposes them through ``get_num_real_samples`` /
``drop_pad_segment``) and give the number of padded rows as ``pad_tokens`` when the packed tensor still carries
them. Splitting skips those rows; merging writes them as zeros.

Example:
    .. code-block:: python

        import torch

        from colossalai.shardformer.modeling.cosmos_mot import layout_from_split_lengths

        # Two samples: [3 AR | 5 DM] and [2 AR | 4 DM], in that pack order.
        layout = layout_from_split_lengths([3, 5, 2, 4], ["causal", "full", "causal", "full"], [8, 6])
        packed = torch.arange(layout.sequence_length)
        ar_block, dm_block = layout.split_tensor(packed)
        assert ar_block.tolist() == [0, 1, 2, 8, 9]
        assert dm_block.tolist() == [3, 4, 5, 6, 7, 10, 11, 12, 13]
        assert layout.merge_outputs(ar_block, dm_block).tolist() == packed.tolist()
"""

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from ..layer.two_way_attn import TwoWayPackedMeta

__all__ = ["CosmosTwoWayLayout", "layout_from_split_lengths", "layout_from_split_info"]

_MISSING = object()
_SUPPORTED_ATTN_MODES = ("causal", "full")


@dataclass(frozen=True)
class CosmosTwoWayLayout:
    """A Cosmos MoT pack in the two-way flat-attention layout.

    The pack order is per sample ``[AR_i | DM_i]``; the attention layout is the global AR block followed by the
    global DM block. ``pad_tokens`` trailing rows belong to no sample: they are skipped by
    :meth:`split_tensor` and written as zeros by :meth:`merge_outputs`.

    Args:
        meta (TwoWayPackedMeta): Per-sample AR/DM lengths in batch order.
        pad_tokens (int, optional): Number of trailing padding rows of the packed tensor. Defaults to 0.

    Raises:
        TypeError: If ``pad_tokens`` is not an integer.
        ValueError: If ``pad_tokens`` is negative, or if a sample contributes no token at all.
    """

    meta: TwoWayPackedMeta
    pad_tokens: int = 0

    def __post_init__(self) -> None:
        pad_tokens = operator.index(self.pad_tokens)
        if pad_tokens < 0:
            raise ValueError(f"pad_tokens must be non-negative, got {pad_tokens}.")
        object.__setattr__(self, "pad_tokens", pad_tokens)
        empty = [
            index
            for index, (ar_len, dm_len) in enumerate(zip(self.meta.ar_lens, self.meta.dm_lens))
            if ar_len + dm_len == 0
        ]
        if empty:
            raise ValueError(
                f"Sample(s) {empty} contribute no token, which a cumulative-length segment cannot describe. Drop them "
                f"before building the layout."
            )

    @property
    def num_samples(self) -> int:
        """Number of real samples in the batch."""
        return self.meta.num_samples

    @property
    def real_tokens(self) -> int:
        """Number of tokens that belong to a sample, i.e. the packed length without padding."""
        return self.meta.total_kv_tokens

    @property
    def sequence_length(self) -> int:
        """Length of the packed tensor, padding included."""
        return self.real_tokens + self.pad_tokens

    def split_tensor(self, packed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split a pack-order tensor into the global AR and DM blocks of the kernel.

        Args:
            packed (torch.Tensor): Tensor of shape ``[sequence_length, ...]`` whose rows follow the per-sample
                ``[AR_i | DM_i]`` pack order, padding included.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: AR block of shape ``[total_ar_tokens, ...]`` and DM block of shape
            ``[total_dm_tokens, ...]``.

        Raises:
            ValueError: If ``packed`` does not have ``sequence_length`` rows.
        """
        if packed.dim() == 0:
            raise ValueError("packed must have at least one dimension, got a scalar tensor.")
        if packed.size(0) != self.sequence_length:
            raise ValueError(
                f"packed holds {packed.size(0)} rows, but the layout describes {self.sequence_length} "
                f"({self.real_tokens} real tokens and {self.pad_tokens} padded rows)."
            )

        tail = tuple(packed.shape[1:])
        ar_parts: List[torch.Tensor] = []
        dm_parts: List[torch.Tensor] = []
        offset = 0
        for ar_len, dm_len in zip(self.meta.ar_lens, self.meta.dm_lens):
            if ar_len > 0:
                ar_parts.append(packed[offset : offset + ar_len])
                offset += ar_len
            if dm_len > 0:
                dm_parts.append(packed[offset : offset + dm_len])
                offset += dm_len

        ar_block = torch.cat(ar_parts, dim=0) if ar_parts else packed.new_empty((0,) + tail)
        dm_block = torch.cat(dm_parts, dim=0) if dm_parts else packed.new_empty((0,) + tail)
        return ar_block, dm_block

    def merge_outputs(self, out_ar: torch.Tensor, out_dm: torch.Tensor) -> torch.Tensor:
        """Put the kernel's two outputs back into pack order, zero-filling the pad tail.

        Args:
            out_ar (torch.Tensor): AR output of shape ``[total_ar_tokens, ...]``.
            out_dm (torch.Tensor): DM output of shape ``[total_dm_tokens, ...]``.

        Returns:
            torch.Tensor: Packed-order tensor of shape ``[sequence_length, ...]``. The trailing ``pad_tokens`` rows
            are zero.

        Raises:
            ValueError: If the two outputs do not match the metadata, or if they disagree on the trailing dimensions.
        """
        if out_ar.size(0) != self.meta.total_ar_tokens:
            raise ValueError(
                f"out_ar holds {out_ar.size(0)} tokens, but the layout describes {self.meta.total_ar_tokens}."
            )
        if out_dm.size(0) != self.meta.total_dm_tokens:
            raise ValueError(
                f"out_dm holds {out_dm.size(0)} tokens, but the layout describes {self.meta.total_dm_tokens}."
            )
        if out_ar.shape[1:] != out_dm.shape[1:]:
            raise ValueError(
                f"out_ar and out_dm must agree on the trailing dimensions, got {tuple(out_ar.shape[1:])} and "
                f"{tuple(out_dm.shape[1:])}."
            )

        parts: List[torch.Tensor] = []
        ar_offset = 0
        dm_offset = 0
        for ar_len, dm_len in zip(self.meta.ar_lens, self.meta.dm_lens):
            if ar_len > 0:
                parts.append(out_ar[ar_offset : ar_offset + ar_len])
                ar_offset += ar_len
            if dm_len > 0:
                parts.append(out_dm[dm_offset : dm_offset + dm_len])
                dm_offset += dm_len
        if self.pad_tokens > 0:
            parts.append(out_ar.new_zeros((self.pad_tokens,) + tuple(out_ar.shape[1:])))

        if not parts:
            return out_ar.new_zeros((self.sequence_length,) + tuple(out_ar.shape[1:]))
        return torch.cat(parts, dim=0)


def _read_field(source, name: str):
    if isinstance(source, Mapping):
        return source.get(name, _MISSING)
    return getattr(source, name, _MISSING)


def layout_from_split_lengths(
    split_lens: Sequence[int],
    attn_modes: Sequence[str],
    sample_lens: Sequence[int],
    *,
    pad_tokens: int = 0,
) -> CosmosTwoWayLayout:
    """Build the layout from a Cosmos packer's split description.

    ``split_lens``/``attn_modes`` describe the attention splits in pack order, and ``sample_lens`` groups them by
    sample: the first ``sample_lens[0]`` tokens are sample 0's splits, and so on. The expected shape of one sample is
    ``["causal", "full"]`` in that order; a sample with only one of the two towers is allowed. Zero-length splits are
    allowed, but a sample whose splits add up to zero is rejected because the kernel cannot describe it.

    Args:
        split_lens (Sequence[int]): Token count of every split, in pack order.
        attn_modes (Sequence[str]): ``"causal"`` or ``"full"`` for every split, in pack order.
        sample_lens (Sequence[int]): Token count of every sample, in batch order.
        pad_tokens (int, optional): Number of trailing padded rows of the packed tensor. Defaults to 0.

    Returns:
        CosmosTwoWayLayout: The layout of the batch.

    Raises:
        TypeError: If a length is not an integer.
        ValueError: If the descriptions disagree, if a sample has more than one split of a mode, if a full split
            precedes the causal one, or if a sample contributes no token.

    Example:
        .. code-block:: python

            from colossalai.shardformer.modeling.cosmos_mot import layout_from_split_lengths

            layout = layout_from_split_lengths([3, 5], ["causal", "full"], [8])
            assert layout.meta.ar_lens == (3,)
            assert layout.meta.dm_lens == (5,)
    """
    split_lengths = [operator.index(length) for length in split_lens]
    sample_lengths = [operator.index(length) for length in sample_lens]
    modes = [str(mode).lower() for mode in attn_modes]

    if len(split_lengths) != len(modes):
        raise ValueError(
            f"split_lens and attn_modes must describe the same number of splits, got {len(split_lengths)} and "
            f"{len(modes)}."
        )
    if any(length < 0 for length in split_lengths):
        raise ValueError(f"Every split length must be non-negative, got split_lens={split_lengths}.")
    if any(length < 0 for length in sample_lengths):
        raise ValueError(f"Every sample length must be non-negative, got sample_lens={sample_lengths}.")
    if len(sample_lengths) == 0:
        raise ValueError("sample_lens must describe at least one sample.")
    if sum(split_lengths) != sum(sample_lengths):
        raise ValueError(
            f"split_lens sum to {sum(split_lengths)}, but sample_lens sum to {sum(sample_lengths)}. Both must "
            f"describe the same packed tokens."
        )

    ar_lens: List[int] = []
    dm_lens: List[int] = []
    cursor = 0
    for sample_index, sample_len in enumerate(sample_lengths):
        if sample_len == 0:
            raise ValueError(
                f"Sample {sample_index} contributes no token, which a cumulative-length segment cannot describe. Drop "
                f"it before building the layout."
            )
        ar_len = 0
        dm_len = 0
        seen: List[str] = []
        consumed = 0
        while consumed < sample_len:
            if cursor == len(split_lengths):
                raise ValueError(
                    f"Sample {sample_index} declares {sample_len} tokens, but the splits run out after " f"{consumed}."
                )
            mode = modes[cursor]
            if mode not in _SUPPORTED_ATTN_MODES:
                raise ValueError(
                    f"Unsupported attention mode {attn_modes[cursor]!r} at split {cursor}; expected one of "
                    f"{list(_SUPPORTED_ATTN_MODES)}. The two-way layout has no third mode."
                )
            if mode in seen:
                raise ValueError(
                    f"Sample {sample_index} has more than one {mode!r} split. The two-way layout expects at most one "
                    f"causal split and one full split per sample."
                )
            if mode == "causal" and "full" in seen:
                raise ValueError(
                    f"Sample {sample_index} lists a full split before its causal split. The pack order must be "
                    f"[causal, full]."
                )
            length = split_lengths[cursor]
            if mode == "causal":
                ar_len += length
            else:
                dm_len += length
            seen.append(mode)
            consumed += length
            cursor += 1
        if consumed != sample_len:
            raise ValueError(
                f"Sample {sample_index} declares {sample_len} tokens, but its splits add up to {consumed}."
            )
        ar_lens.append(ar_len)
        dm_lens.append(dm_len)

    if cursor != len(split_lengths) and any(length > 0 for length in split_lengths[cursor:]):
        raise ValueError(
            f"{len(split_lengths) - cursor} split(s) remain after the last sample. sample_lens and split_lens must "
            f"describe the same packed tokens."
        )

    return CosmosTwoWayLayout(TwoWayPackedMeta(ar_lens, dm_lens), pad_tokens=pad_tokens)


def layout_from_split_info(
    split_info,
    *,
    sample_lens: Optional[Sequence[int]] = None,
    pad_tokens: int = 0,
) -> CosmosTwoWayLayout:
    """Build the layout from a Cosmos ``SplitInfo``/``SequencePack``-like object.

    The object may be a mapping or carry the fields as attributes; the Cosmos packer's ``SplitInfo`` and
    ``SequencePack`` do both for these three fields.

    Args:
        split_info: Object carrying ``split_lens``, ``attn_modes`` and, unless overridden, ``sample_lens``.
        sample_lens (Optional[Sequence[int]], optional): Overrides the object's ``sample_lens``. Defaults to None.
        pad_tokens (int, optional): Number of trailing padded rows of the packed tensor. Defaults to 0.

    Returns:
        CosmosTwoWayLayout: The layout of the batch.

    Raises:
        TypeError: If the object does not carry ``split_lens``, ``attn_modes`` or ``sample_lens``.
        ValueError: If the descriptions disagree, as in :func:`layout_from_split_lengths`.
    """
    split_lens = _read_field(split_info, "split_lens")
    attn_modes = _read_field(split_info, "attn_modes")
    if sample_lens is None:
        sample_lens = _read_field(split_info, "sample_lens")

    missing = [
        name
        for name, value in (("split_lens", split_lens), ("attn_modes", attn_modes), ("sample_lens", sample_lens))
        if value is _MISSING or value is None
    ]
    if missing:
        raise TypeError(
            f"{type(split_info).__name__} does not carry {missing}. A Cosmos SplitInfo or SequencePack has all "
            f"three fields; pass sample_lens explicitly for a partial object."
        )

    return layout_from_split_lengths(split_lens, attn_modes, sample_lens, pad_tokens=pad_tokens)

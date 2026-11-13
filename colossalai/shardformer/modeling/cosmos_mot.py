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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..layer.two_way_attn import TwoWayPackedMeta

__all__ = [
    "CosmosTwoWayLayout",
    "Cosmos3PackedBatch",
    "layout_from_split_lengths",
    "layout_from_split_info",
    "pack_cosmos3_samples",
    "token_mean_loss_scale",
    "two_way_varlen_summary",
    "all_reduce_supervised_tokens",
]

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


# The packer owns the per-sample field names. These sets are the contract between a Cosmos 3 sample dictionary and
# the global ``[AR_all; DM_all]`` tensor that the transformer forward consumes.
_AR_FIELD_NAMES = ("input_ids", "text_indexes")
_AR_PREFIXES = ("text_",)
_PER_SAMPLE_LIST_SUFFIXES = (
    "_tokens",
    "_token_shapes",
    "_noisy_frame_indexes",
    "_domain_ids",
)
_PER_SAMPLE_LIST_FIELDS = ("original_latent_shapes",)
_DM_INDEX_SUFFIXES = ("_sequence_indexes", "_mse_loss_indexes")
_DM_VALUE_FIELDS = ("vision_timesteps", "action_timesteps", "sound_timesteps", "audio_timesteps")


@dataclass(frozen=True)
class Cosmos3PackedBatch:
    """A batch of Cosmos 3 samples packed into the two-way layout.

    ``model_inputs`` is ready to feed to ``Cosmos3OmniTransformer.forward``. Every tensor and index points at the
    global ``[AR_all; DM_all]`` sequence, so a model call needs no further reordering. ``layout`` carries the
    attention metadata; ``supervised_tokens`` records, per loss family, how many tokens each sample contributes, which
    is the denominator a token-mean loss needs across a data-parallel group.

    Args:
        model_inputs (Dict[str, Any]): The packed keyword arguments of the transformer forward.
        layout (CosmosTwoWayLayout): The pack layout, padding included.
        sample_ids (Tuple[str, ...]): Stable identifier of every sample, in batch order.
        supervised_tokens (Dict[str, Tuple[int, ...]]): Per-family supervised token count of every sample. The
            ``"total"`` entry is the sum over families.
    """

    model_inputs: Dict[str, Any]
    layout: CosmosTwoWayLayout
    sample_ids: Tuple[str, ...]
    supervised_tokens: Dict[str, Tuple[int, ...]]

    @property
    def meta(self) -> TwoWayPackedMeta:
        """The attention metadata of the batch."""
        return self.layout.meta

    @property
    def total_supervised_tokens(self) -> int:
        """Number of supervised tokens in the local pack."""
        return sum(self.supervised_tokens["total"])


def _field_kind(name: str) -> str:
    """Classify a sample field by where it belongs in the global sequence."""
    if name in _AR_FIELD_NAMES:
        return "ar"
    if name == "position_ids":
        return "split"
    if name.endswith(_PER_SAMPLE_LIST_SUFFIXES) or name in _PER_SAMPLE_LIST_FIELDS:
        return "list"
    if name.endswith(_DM_INDEX_SUFFIXES) or name.startswith(("vision_", "action_", "sound_", "audio_")):
        return "dm"
    raise ValueError(
        f"Unknown Cosmos 3 sample field {name!r}. Add it to the packing contract in "
        f"colossalai/shardformer/modeling/cosmos_mot.py: AR fields go to _AR_FIELD_NAMES, per-sample lists end with "
        f"one of {_PER_SAMPLE_LIST_SUFFIXES}, and diffusion fields use a vision_/action_/sound_/audio_ prefix or an "
        f"index suffix."
    )


def _shift_dm_indexes(indexes: torch.Tensor, shift: int) -> torch.Tensor:
    """Move a sample's DM indexes from its local ``[AR_i | DM_i]`` offset to the global DM block."""
    if shift == 0:
        return indexes
    return indexes + shift


def pack_cosmos3_samples(samples: Sequence[Dict], *, pad_tokens: int = 0) -> Cosmos3PackedBatch:
    """Pack per-sample dictionaries into one global ``[AR_all; DM_all]`` transformer input.

    Every sample must describe its own ``[AR_i | DM_i]`` pack with ``und_len`` and ``sequence_length``. The merged
    batch has the global AR block first and the global DM block second, which is the layout
    :class:`TwoWayPackedMeta` expects. All indexes are rewritten in the same pass:

    * ``input_ids`` and ``text_indexes`` stay in the AR block;
    * ``position_ids`` is split per sample, so each sample keeps its own semantic coordinates;
    * ``vision_*`` / ``action_*`` / ``sound_*`` / ``audio_*`` indexes move to the global DM block;
    * per-sample list fields (tokens, shapes, noisy frames, domain ids) are concatenated in batch order.

    Padding is recorded in the layout and zero-fills the packed tensors. It belongs to no sample, so it enters
    neither the attention segments nor the supervised token count.

    Args:
        samples (Sequence[Dict]): Per-sample packer outputs. Every sample must carry the same field names.
        pad_tokens (int, optional): Number of trailing rows that belong to no sample. Defaults to 0.

    Returns:
        Cosmos3PackedBatch: The packed inputs, layout, sample ids and supervised token counts.

    Raises:
        ValueError: If a sample is missing ``und_len``/``sequence_length``, if the field sets disagree, if a field
            cannot be classified, or if the layout rejects the lengths.

    Example:
        .. code-block:: python

            from colossalai.shardformer.modeling.cosmos_mot import pack_cosmos3_samples

            batch = pack_cosmos3_samples(
                [
                    {"und_len": 3, "sequence_length": 8, "input_ids": ids_a, "text_indexes": text_a},
                    {"und_len": 2, "sequence_length": 6, "input_ids": ids_b, "text_indexes": text_b},
                ]
            )
            assert batch.meta.ar_lens == (3, 2)
            assert batch.meta.dm_lens == (5, 4)
    """
    if not samples:
        raise ValueError("pack_cosmos3_samples needs at least one sample.")
    required = ("und_len", "sequence_length")
    field_sets = []
    ar_lens: List[int] = []
    dm_lens: List[int] = []
    sample_ids: List[str] = []
    for index, sample in enumerate(samples):
        missing = [name for name in required if name not in sample]
        if missing:
            raise ValueError(f"Sample {index} is missing {missing}; every packer output needs both fields.")
        und_len = operator.index(sample["und_len"])
        sequence_length = operator.index(sample["sequence_length"])
        if not 0 <= und_len <= sequence_length:
            raise ValueError(
                f"Sample {index} has und_len={und_len} and sequence_length={sequence_length}; und_len must be in "
                f"[0, sequence_length]."
            )
        ar_lens.append(und_len)
        dm_lens.append(sequence_length - und_len)
        sample_ids.append(str(sample.get("sample_id", index)))
        field_sets.append(tuple(sorted(name for name in sample if name not in required and name != "sample_id")))

    if len(set(field_sets)) != 1:
        raise ValueError(
            f"Every sample must carry the same packer fields, got {sorted(set(field_sets))}. A missing field would "
            f"silently shift the global indexes; pad it explicitly instead."
        )
    fields = field_sets[0]
    layout = CosmosTwoWayLayout(TwoWayPackedMeta(ar_lens, dm_lens), pad_tokens=pad_tokens)
    total_ar = layout.meta.total_ar_tokens

    merged: Dict[str, object] = {}
    supervised: Dict[str, List[int]] = {}
    for name in fields:
        kind = _field_kind(name)
        values = [sample[name] for sample in samples]
        if kind == "list":
            flattened = []
            for value in values:
                if not isinstance(value, (list, tuple)):
                    raise ValueError(f"{name} must be a list per sample, got {type(value).__name__}.")
                flattened.extend(value)
            merged[name] = flattened
            continue

        if kind == "split":
            leading = values[0].shape[:-1]
            if any(value.shape[:-1] != leading for value in values):
                raise ValueError(f"Every {name} tensor must agree on its leading dimensions.")
            ar_parts = [value[..., :und_len] for value, und_len in zip(values, ar_lens)]
            dm_parts = [value[..., und_len:] for value, und_len in zip(values, ar_lens)]
            merged[name] = torch.cat(ar_parts + dm_parts, dim=-1)
            continue

        ar_offset = 0
        dm_offset = total_ar
        parts = []
        for sample, value, ar_len, dm_len in zip(samples, values, ar_lens, dm_lens):
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"{name} must be a tensor per sample, got {type(value).__name__}.")
            if kind == "ar":
                # `input_ids` carries vocabulary ids; every other AR field is an index into the global sequence and
                # must follow its sample's position in the AR block.
                parts.append(value if name == "input_ids" else _shift_dm_indexes(value, ar_offset))
                ar_offset += ar_len
            else:
                # Diffusion timesteps are values, not positions, so they are concatenated without an offset.
                if name in _DM_VALUE_FIELDS:
                    parts.append(value)
                else:
                    shift = dm_offset - ar_len
                    parts.append(_shift_dm_indexes(value, shift))
                dm_offset += dm_len
        merged[name] = torch.cat(parts) if parts else values[0].new_empty((0,))

    for name in fields:
        if name.endswith("_mse_loss_indexes"):
            supervised[name] = [int(sample[name].numel()) for sample in samples]
    total_per_sample = [0] * len(samples)
    for counts in supervised.values():
        for index, count in enumerate(counts):
            total_per_sample[index] += count

    merged.update(
        und_len=total_ar,
        # The model consumes real tokens only; the padded tail lives in `layout` for a packed kernel that still
        # carries it. Keeping the model's sequence length free of padding prevents a position/RoPE over-read.
        sequence_length=layout.real_tokens,
    )
    supervised["total"] = tuple(total_per_sample)
    return Cosmos3PackedBatch(
        model_inputs=merged,
        layout=layout,
        sample_ids=tuple(sample_ids),
        supervised_tokens={name: tuple(counts) for name, counts in supervised.items()},
    )


def token_mean_loss_scale(
    local_supervised_tokens: int,
    global_supervised_tokens: int,
    data_parallel_size: int,
) -> float:
    """Multiplier that turns a local mean loss into a global token mean under data-parallel gradient averaging.

    ZeRO averages gradients across the data-parallel group. If every rank contributed its own token mean, ranks with
    short packs would weigh as much as ranks with long packs. Multiplying the local mean by
    ``local_tokens * data_parallel_size / global_tokens`` makes that average equal the mean over all supervised
    tokens of the step.

    With gradient accumulation, pass the token counts and the mean loss of the current microbatch and take
    ``global_supervised_tokens`` over the whole accumulation window. The caller is responsible for the same window on
    every rank.

    Args:
        local_supervised_tokens (int): Supervised tokens in this rank's microbatch.
        global_supervised_tokens (int): Supervised tokens over all ranks and all accumulated microbatches.
        data_parallel_size (int): Size of the data-parallel group whose gradients are averaged.

    Returns:
        float: Scale to apply to the local mean loss before ``backward``.

    Raises:
        ValueError: If a count is negative, the group is empty, or the global count is zero.

    Example:
        .. code-block:: python

            from colossalai.shardformer.modeling.cosmos_mot import token_mean_loss_scale

            # Rank 0 has 10 supervised tokens, rank 1 has 30; the global mean weights them 1:3.
            assert token_mean_loss_scale(10, 40, 2) == 0.5
            assert token_mean_loss_scale(30, 40, 2) == 1.5
    """
    if data_parallel_size <= 0:
        raise ValueError(f"data_parallel_size must be positive, got {data_parallel_size}.")
    if local_supervised_tokens < 0 or global_supervised_tokens < 0:
        raise ValueError(
            f"Token counts must be non-negative, got local={local_supervised_tokens} and "
            f"global={global_supervised_tokens}."
        )
    if global_supervised_tokens == 0:
        raise ValueError("The accumulation window has no supervised token; cannot normalise a token mean loss.")
    return local_supervised_tokens * data_parallel_size / global_supervised_tokens


def two_way_varlen_summary(meta: TwoWayPackedMeta) -> Dict[str, object]:
    """Return the cumulative lengths a packed kernel call would receive, for logs and benchmark manifests.

    The lists are host values; they are the same prefix sums the CUDA/NPU backends turn into device tensors.
    """
    return {
        "num_samples": meta.num_samples,
        "total_ar_tokens": meta.total_ar_tokens,
        "total_dm_tokens": meta.total_dm_tokens,
        "max_ar": meta.max_ar,
        "max_dm": meta.max_dm,
        "max_kv": meta.max_kv,
        "cu_seqlens_ar": meta.cu_seqlens_ar,
        "cu_seqlens_dm": meta.cu_seqlens_dm,
        "cu_seqlens_kv": meta.cu_seqlens_kv,
    }


def all_reduce_supervised_tokens(local_tokens: int, group=None, device: Optional[torch.device] = None) -> int:
    """Sum supervised token counts over a process group, exactly, on the accelerator's device.

    The reduction tensor is int64: a token count is exact there, and both NCCL and HCCL support int64 collectives.
    A float64 temporary would not work on Ascend -- HCCL has no float64 all-reduce -- and the default device must
    be the accelerator's rather than the host's, or the collective would have no backend to run on.

    Args:
        local_tokens (int): Supervised tokens on this rank.
        group (optional): Process group to reduce over. Defaults to the default group.
        device (Optional[torch.device], optional): Device of the temporary reduction tensor. Defaults to the current
            accelerator device.

    Returns:
        int: Sum over the group.

    Raises:
        RuntimeError: If distributed is not initialised.
    """
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("all_reduce_supervised_tokens requires an initialised torch.distributed process group.")
    if device is None:
        from colossalai.accelerator import get_accelerator

        device = get_accelerator().get_current_device()
    total = torch.tensor([int(local_tokens)], dtype=torch.int64, device=device)
    dist.all_reduce(total, group=group)
    return int(total.item())

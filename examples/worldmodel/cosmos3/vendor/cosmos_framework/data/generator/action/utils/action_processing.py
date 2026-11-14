# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared action processing records and normalization helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch

from cosmos_framework.utils import log

ActionNormalizationMethod = Literal[
    "quantile",
    "quantile_rot",
    "quantile_rot_scale_floor",
    "asinh_rot",
    "piecewise_asinh_rot",
    "global_asinh",
    "meanstd",
    "minmax",
]
QUANTILE_ROT_SCALE_FLOOR_TARGET_MAX = 5.0
_RAW_ROTATION_STATS_METHODS = frozenset(
    {"quantile_rot", "quantile_rot_scale_floor", "asinh_rot", "piecewise_asinh_rot"}
)
_GLOBAL_ASINH_POSE_CONVENTIONS = frozenset(
    {"backward_anchored", "backward_chunk_anchored_8f", "backward_chunk_anchored_16f"}
)
GLOBAL_ASINH_GRIPPER_01_STATS_PATH = (
    Path(__file__).parents[1] / "normalizers" / "global_asinh_v1_backward_anchored_gripper_binary.json"
)
GLOBAL_ASINH_METRIC_GRIPPER_STATS_PATH = (
    Path(__file__).parents[1] / "normalizers" / "global_asinh_v1_backward_anchored_gripper_metric.json"
)
ActionNormalizationStats = dict[str, torch.Tensor]


class ActionNormalizer(Protocol):
    """Tensor-level action normalization interface used by ActionProcessor."""

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:  # action: [...,D], returns [...,D]
        """Map raw action values into model-space values."""
        ...

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:  # action: [...,D], returns [...,D]
        """Invert model-space action values back into raw action values."""
        ...


class ActionCodec(Protocol):
    """Decode a canonical normalized action into canonical raw space."""

    def decode_to_unified(
        self, normalized_action: torch.Tensor
    ) -> torch.Tensor:  # normalized_action: [...,D], returns [...,D]
        """Decode normalized values into the canonical raw action layout."""
        ...


@dataclass(frozen=True)
class ActionAffineNormalization:
    """Resolved affine action normalizer.

    Forward normalization is ``(raw - offset) / scale``.  Inverse
    denormalization is ``normalized * scale + offset``.

    ``forward_clamp`` records lossy range-style forward clamping.  When
    ``forward_clamp_mask`` is provided, only channels with a ``True`` mask
    entry are clamped; this represents mixed UMI normalizers where some fields
    are range-clamped and others are plain affine transforms.
    """

    offset: torch.Tensor
    scale: torch.Tensor
    forward_clamp: tuple[float, float] | None = None
    forward_clamp_mask: torch.Tensor | None = None

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:  # action: [...,D], returns [...,D]
        """Normalize raw action values with resolved affine parameters."""
        offset = self.offset.to(device=action.device, dtype=action.dtype)  # [D]
        scale = self.scale.to(device=action.device, dtype=action.dtype)  # [D]
        normalized = (action - offset) / scale  # [...,D]
        if self.forward_clamp is not None:
            lo, hi = self.forward_clamp
            clamped = normalized.clamp(lo, hi)  # [...,D]
            if self.forward_clamp_mask is None:
                normalized = clamped  # [...,D]
            else:
                clamp_mask = self.forward_clamp_mask.to(device=action.device, dtype=torch.bool)  # [D]
                normalized = torch.where(clamp_mask, clamped, normalized)  # [...,D]
        return normalized  # [...,D]

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:  # action: [...,D], returns [...,D]
        """Invert action normalization with resolved affine parameters."""
        offset = self.offset.to(device=action.device, dtype=action.dtype)  # [D]
        scale = self.scale.to(device=action.device, dtype=action.dtype)  # [D]
        return action * scale + offset  # [...,D]


@dataclass(frozen=True)
class ActionAsinhNormalization:
    """Invertible heavy-tail action normalizer.

    The forward transform first applies the same quantile affine transform used
    by ``quantile_rot`` and then compresses large-but-valid tails with
    ``asinh``:

    ``y = asinh((raw - offset) / scale) / asinh(1)``.

    The ``asinh(1)`` factor keeps q01/q99-style unit deviations near ±1 while
    mapping 100-250× outliers to single-digit targets instead of exploding the
    flow-matching loss.  The inverse uses ``sinh`` and is exact up to floating
    point precision; unlike clamp, no labels are censored.
    """

    offset: torch.Tensor
    scale: torch.Tensor
    unit: float = math.asinh(1.0)
    action_schema: Literal["native", "unified_v1"] = "native"

    def _validate_width(self, action: torch.Tensor) -> None:  # action: [...,D]
        if action.ndim == 0 or action.shape[-1] != self.offset.numel():
            raise ValueError(
                f"action width {action.shape[-1] if action.ndim else 0} does not match normalizer width "
                f"{self.offset.numel()}"
            )

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:  # action: [...,D], returns [...,D]
        """Normalize raw action values with an invertible asinh transform."""
        self._validate_width(action)
        offset = self.offset.to(device=action.device, dtype=action.dtype)  # [D]
        scale = self.scale.to(device=action.device, dtype=action.dtype)  # [D]
        unit = torch.as_tensor(self.unit, device=action.device, dtype=action.dtype)  # []
        return torch.asinh((action - offset) / scale) / unit  # [...,D]

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:  # action: [...,D], returns [...,D]
        """Invert asinh action normalization."""
        self._validate_width(action)
        offset = self.offset.to(device=action.device, dtype=action.dtype)  # [D]
        scale = self.scale.to(device=action.device, dtype=action.dtype)  # [D]
        unit = torch.as_tensor(self.unit, device=action.device, dtype=action.dtype)  # []
        return torch.sinh(action * unit) * scale + offset  # [...,D]


@dataclass(frozen=True)
class ActionPiecewiseAsinhNormalization:
    """Invertible tail-only asinh action normalizer.

    This uses the same affine ``z = (raw - offset) / scale`` coordinates as
    ``quantile_rot`` and keeps the common range exactly linear:

    ``y = z`` when ``|z| <= 1``.

    Outside that range it grows sublinearly while preserving magnitude order:

    ``y = sign(z) * (1 + asinh(beta * (|z| - 1)) / beta)``.

    With the default ``beta=1``, the transform is continuous and has matching
    derivative at ±1, so in-range samples match the linear/clamped normalizer
    exactly while out-of-range labels remain recoverable instead of censored.
    """

    offset: torch.Tensor
    scale: torch.Tensor
    beta: float = 1.0

    def _beta_tensor(self, reference: torch.Tensor) -> torch.Tensor:
        beta = torch.as_tensor(self.beta, device=reference.device, dtype=reference.dtype)  # []
        if bool((beta <= 0).item()):
            raise ValueError(f"piecewise asinh beta must be positive, got {self.beta}")
        return beta

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:  # action: [...,D], returns [...,D]
        """Normalize raw action values, keeping affine-normalized [-1, 1] exactly unchanged."""
        offset = self.offset.to(device=action.device, dtype=action.dtype)  # [D]
        scale = self.scale.to(device=action.device, dtype=action.dtype)  # [D]
        beta = self._beta_tensor(action)  # []
        z = (action - offset) / scale  # [...,D]
        abs_z = z.abs()  # [...,D]
        tail = 1.0 + torch.asinh(beta * (abs_z - 1.0)) / beta  # [...,D]
        return torch.where(abs_z <= 1.0, z, z.sign() * tail)  # [...,D]

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:  # action: [...,D], returns [...,D]
        """Invert piecewise-asinh action normalization."""
        offset = self.offset.to(device=action.device, dtype=action.dtype)  # [D]
        scale = self.scale.to(device=action.device, dtype=action.dtype)  # [D]
        beta = self._beta_tensor(action)  # []
        abs_y = action.abs()  # [...,D]
        tail = 1.0 + torch.sinh(beta * (abs_y - 1.0)) / beta  # [...,D]
        z = torch.where(abs_y <= 1.0, action, action.sign() * tail)  # [...,D]
        return z * scale + offset  # [...,D]


def load_action_stats(stats_path: str, stats_key: str = "global") -> dict[str, np.ndarray]:
    """Load pre-computed action normalization stats from a JSON file."""
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Action normalization stats not found at {stats_path}.")
    log.info(f"Loading action normalization stats from {stats_path}")
    with path.open("r") as f:
        raw = json.load(f)
    stat_keys = {"mean", "std", "min", "max", "q01", "q99"}
    if stats_key in raw:
        raw = raw[stats_key]
        if not isinstance(raw, dict):
            raise TypeError(f"Action normalization stats block {stats_key!r} in {stats_path} must be a dict.")
    elif stats_key != "global" and not any(key in raw for key in stat_keys):
        raise KeyError(f"Action normalization stats block {stats_key!r} not found in {stats_path}.")
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items() if k in stat_keys}


def resolve_action_normalization(
    method: ActionNormalizationMethod,
    stats: ActionNormalizationStats | None = None,
    apply_forward_clamp: bool = False,
    *,
    pose_convention: str | None = None,
) -> ActionNormalizer:
    """Resolve one configured normalization method into its forward/inverse transform."""
    if stats is None:
        raise ValueError(f"{method} normalization requires per-dataset action statistics")
    if method == "global_asinh" and pose_convention not in _GLOBAL_ASINH_POSE_CONVENTIONS:
        raise ValueError(f"global_asinh normalization requires an anchored pose convention, got {pose_convention!r}")

    if method == "meanstd":
        offset = stats["mean"]  # [D]
        scale = stats["std"].clamp(min=1e-8)  # [D]
        return ActionAffineNormalization(offset=offset, scale=scale)

    if method == "minmax":
        lo = stats["min"]  # [D]
        hi = stats["max"]  # [D]
    elif method in (
        "quantile",
        "quantile_rot",
        "quantile_rot_scale_floor",
        "asinh_rot",
        "piecewise_asinh_rot",
        "global_asinh",
    ):
        lo = stats["q01"]  # [D]
        hi = stats["q99"]  # [D]
    else:
        raise ValueError(f"Unknown normalization method: {method!r}")

    offset = (hi + lo) / 2.0  # [D]
    scale = (hi - lo).clamp(min=1e-8) / 2.0  # [D]
    if method == "quantile_rot_scale_floor":
        min_dev = (stats["min"] - offset).abs()  # [D]
        max_dev = (stats["max"] - offset).abs()  # [D]
        scale_floor = torch.maximum(min_dev, max_dev) / QUANTILE_ROT_SCALE_FLOOR_TARGET_MAX  # [D]
        scale = torch.maximum(scale, scale_floor.clamp(min=1e-8))  # [D]

    if method in {"asinh_rot", "global_asinh"}:
        action_schema = "unified_v1" if method == "global_asinh" else "native"
        return ActionAsinhNormalization(offset=offset, scale=scale, action_schema=action_schema)
    if method == "piecewise_asinh_rot":
        return ActionPiecewiseAsinhNormalization(offset=offset, scale=scale)

    if apply_forward_clamp:
        # Ideally this hardcode should be removed, but for now we keep it so we can be aligned with mid-training checkpoints.
        forward_clamp = (-1.0, 1.0)
    else:
        forward_clamp = None

    return ActionAffineNormalization(
        offset=offset,
        scale=scale,
        forward_clamp=forward_clamp,
    )


def load_action_normalization_stats(
    action_normalization: ActionNormalizationMethod,
    *,
    stats_path: str | Path,
    stats_key: str | None = None,
    expected_dim: int | None = None,
) -> dict[str, torch.Tensor]:
    """Load tensor action stats for a configured normalization method."""
    if action_normalization == "global_asinh":
        if expected_dim is not None and expected_dim != 59:
            raise ValueError(f"global_asinh stats require expected_dim=59, got {expected_dim}")
        expected_dim = 59
    if stats_key is None:
        stats_key = "global_raw" if action_normalization in _RAW_ROTATION_STATS_METHODS else "global"
    raw_stats = load_action_stats(str(stats_path), stats_key=stats_key)
    if expected_dim is not None:
        for name, values in raw_stats.items():
            if values.shape != (expected_dim,):
                raise ValueError(f"Action stats {name!r} must have shape [{expected_dim}], got {values.shape}.")
    return {name: torch.from_numpy(values).float() for name, values in raw_stats.items()}


def load_action_normalizer(
    action_normalization: ActionNormalizationMethod | None,
    *,
    stats_path: str | Path,
    apply_forward_clamp: bool,
    stats_key: str | None = None,
    expected_dim: int | None = None,
) -> ActionNormalizer | None:
    """Load and resolve an action normalizer from pre-computed stats."""
    if action_normalization is None:
        return None
    return resolve_action_normalization(
        action_normalization,
        load_action_normalization_stats(
            action_normalization,
            stats_path=stats_path,
            stats_key=stats_key,
            expected_dim=expected_dim,
        ),
        apply_forward_clamp=apply_forward_clamp,
    )


def make_camera_global_asinh_normalizer(
    stats_path: str | Path = GLOBAL_ASINH_METRIC_GRIPPER_STATS_PATH,
    *,
    pose_convention: str = "backward_anchored",
) -> ActionAsinhNormalization:
    """Build the native 9D camera normalizer from the canonical global-asinh profile."""
    if pose_convention not in _GLOBAL_ASINH_POSE_CONVENTIONS:
        raise ValueError(f"global_asinh normalization requires an anchored pose convention, got {pose_convention!r}")
    stats = load_action_normalization_stats(
        "global_asinh",
        stats_path=stats_path,
        expected_dim=59,
    )
    lo = stats["q01"][:9]
    hi = stats["q99"][:9]
    return ActionAsinhNormalization(offset=(hi + lo) / 2.0, scale=(hi - lo).clamp(min=1e-8) / 2.0)


def make_pose_action_scale_normalizer(
    action_dim: int,
    *,
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
) -> ActionAffineNormalization:
    """Build a normalizer that maps raw pose deltas into scaled model space.

    Pose actions use the shared layout ``[translation(3), rotation(...)]``.
    The returned normalizer multiplies translation channels by
    ``translation_scale`` and rotation channels by ``rotation_scale`` during
    preprocessing, then inverts those factors during postprocessing.
    """
    if action_dim < 3:
        raise ValueError(f"Pose action_dim must be at least 3, got {action_dim}")
    if translation_scale == 0:
        raise ValueError("translation_scale must be non-zero")
    if rotation_scale == 0:
        raise ValueError("rotation_scale must be non-zero")

    offset = torch.zeros(action_dim, dtype=torch.float32)  # [D]
    scale = torch.ones(action_dim, dtype=torch.float32)  # [D]
    scale[:3] = 1.0 / float(translation_scale)  # [D]
    if action_dim > 3:
        scale[3:] = 1.0 / float(rotation_scale)  # [D]
    return ActionAffineNormalization(offset=offset, scale=scale)


@dataclass(frozen=True)
class ActionProcessingRecord:
    """Per-sample metadata needed to invert Action model-space preprocessing."""

    raw_action_dim: int
    action_normalizer: ActionNormalizer | None
    action_valid_mask: torch.Tensor | None = None  # [D_raw] bool
    action_codec: ActionCodec | None = None

    def __post_init__(self) -> None:
        if self.action_normalizer is not None and self.action_codec is not None:
            raise ValueError("ActionProcessingRecord cannot contain both an action_normalizer and an action_codec")


def pad_action_to_max_dim(
    action: torch.Tensor, max_action_dim: int
) -> torch.Tensor:  # action: [T,D], returns [T,D_model]
    """Pad action tensor to max_action_dim along the last dimension.

    Args:
        action: Action tensor of shape (T, D) where D is the current action dimension.
        max_action_dim: Target action dimension to pad to.

    Returns:
        Padded action tensor of shape (T, max_action_dim).
    """
    if action.shape[-1] > max_action_dim:
        raise ValueError(f"Action dimension {action.shape[-1]} is greater than max_action_dim {max_action_dim}")
    if action.shape[-1] == max_action_dim:
        return action  # [T,D_model]
    padding_size = max_action_dim - action.shape[-1]
    zero_padding = torch.zeros(*action.shape[:-1], padding_size, dtype=action.dtype, device=action.device)  # [T,D_pad]
    return torch.cat([action, zero_padding], dim=-1)  # [T,D_model]


def make_batched_action_processing_fields(
    record: ActionProcessingRecord,
    batch_size: int,
    *,
    action_channel_masking: bool = True,
) -> dict[str, list[torch.Tensor | ActionProcessingRecord | None]]:
    """Build batch-list fields whose action width and inverse record cannot drift apart."""
    raw_action_dim = torch.tensor(record.raw_action_dim, dtype=torch.long) if action_channel_masking else None  # []
    return {
        "raw_action_dim": [raw_action_dim] * batch_size,
        "action_processing_record": [record] * batch_size,
    }


class ActionProcessor:
    """Build model-space actions and decode model outputs for a single sample."""

    def __init__(self, max_action_dim: int, action_channel_masking: bool = True) -> None:
        self.max_action_dim = int(max_action_dim)
        self.action_channel_masking = bool(action_channel_masking)

    def preprocess_action(
        self,
        data_dict: dict[str, Any],
        action: torch.Tensor,
        *,
        action_normalizer: ActionNormalizer | None,
        action_valid_mask: torch.Tensor | None = None,  # [D_raw] bool
        normalized_action: torch.Tensor | None = None,  # [T,D_raw]
        action_codec: ActionCodec | None = None,
    ) -> dict[str, Any]:
        """Return a sample with canonical raw and normalized action fields.

        ``action_raw`` is the exact unnormalized, unpadded action accepted by
        this method. It is the canonical external-space target for evaluation.
        ``action`` is the normalized and padded model-space representation.
        Unified actions may provide that normalized representation explicitly so
        mapping and native normalization stay outside this padding boundary.
        """

        raw_action_dim = int(action.shape[-1])
        if "action_raw" in data_dict:
            raise ValueError("action_raw is reserved for the canonical ActionProcessor input")
        action_raw = action.clone()  # [T,D]
        if action_normalizer is not None and action_codec is not None:
            raise ValueError("Provide either action_normalizer or action_codec, not both")
        if (normalized_action is None) != (action_codec is None):
            raise ValueError("normalized_action and action_codec must be provided together")
        if normalized_action is not None and action_normalizer is not None:
            raise ValueError("normalized_action cannot be combined with action_normalizer")
        if normalized_action is None:
            normalized_action = action  # [T,D]
            if action_normalizer is not None:
                normalized_action = action_normalizer.normalize_action(action)  # [T,D]
        if normalized_action.shape != action.shape:
            raise ValueError(
                f"normalized_action shape {tuple(normalized_action.shape)} does not match "
                f"raw action shape {tuple(action.shape)}"
            )

        if action_valid_mask is not None:
            if action_valid_mask.dtype != torch.bool or action_valid_mask.ndim != 1:
                raise ValueError(
                    f"action_valid_mask must be a 1D bool tensor, got "
                    f"shape={tuple(action_valid_mask.shape)} dtype={action_valid_mask.dtype}"
                )
            if action_valid_mask.numel() != raw_action_dim:
                raise ValueError(
                    f"action_valid_mask width {action_valid_mask.numel()} != raw_action_dim {raw_action_dim}"
                )
            action_valid_mask = action_valid_mask.to(device=normalized_action.device).contiguous()  # [D_raw]

        processed_data_dict = dict(data_dict)
        processed_data_dict["action_raw"] = action_raw
        processed_data_dict["action"] = pad_action_to_max_dim(  # [T,D_model]
            normalized_action, self.max_action_dim
        )
        record = ActionProcessingRecord(
            raw_action_dim=raw_action_dim,
            action_normalizer=action_normalizer,
            action_valid_mask=action_valid_mask,
            action_codec=action_codec,
        )
        processed_data_dict["raw_action_dim"] = (
            torch.tensor(record.raw_action_dim, dtype=torch.long) if self.action_channel_masking else None
        )  # []
        processed_data_dict["action_processing_record"] = record
        if action_valid_mask is not None:
            processed_data_dict["action_valid_mask"] = pad_action_to_max_dim(action_valid_mask, self.max_action_dim).to(
                dtype=torch.bool
            )
        return processed_data_dict

    @staticmethod
    def _unpad_action(action: torch.Tensor, raw_action_dim: int) -> torch.Tensor:
        """Drop model-only padded action channels."""
        if action.shape[-1] < raw_action_dim:
            raise ValueError(f"invalid raw_action_dim={raw_action_dim} for action with shape {tuple(action.shape)}")
        return action[..., :raw_action_dim].contiguous()  # [...,D_raw]

    @staticmethod
    def postprocess_action(
        action: torch.Tensor,
        record: ActionProcessingRecord,
    ) -> torch.Tensor:
        """Unpad and decode a generated model-space action into canonical raw space."""
        action = ActionProcessor._unpad_action(action, record.raw_action_dim)  # [...,D_raw]
        if record.action_codec is not None:
            action = record.action_codec.decode_to_unified(action)  # [...,D_raw]
        elif record.action_normalizer is not None:
            action = record.action_normalizer.denormalize_action(action)  # [...,D_raw]
        return action  # [...,D_raw]

    @staticmethod
    def postprocess_action_to_schema(
        action: torch.Tensor,
        record: ActionProcessingRecord,
    ) -> torch.Tensor:
        """Decode a generated model-space action into its canonical raw schema."""
        return ActionProcessor.postprocess_action(action, record)


def get_action_processing_records(data_batch: dict[str, Any]) -> list[ActionProcessingRecord | None]:
    """Read all per-sample processing records from a collated Action batch."""
    records = data_batch.get("action_processing_record")
    if records is None:
        return []
    if isinstance(records, ActionProcessingRecord):
        return [records]
    if isinstance(records, list):
        for record in records:
            if record is not None and not isinstance(record, ActionProcessingRecord):
                raise TypeError(f"Unexpected action_processing_record entry type: {type(record).__name__}")
        return records
    raise TypeError(f"Unexpected action_processing_record type: {type(records).__name__}")

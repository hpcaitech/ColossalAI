# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared action-vector specification: per-dim type label + idle thresholds.

Single concept: every column of an action vector has a :class:`DimType` label.
Idle detection iterates by type and applies the matching algorithm:

    POS      → ‖action[pos_idx]‖ per arm < eps_t
    ROT      → distance(rot, identity) per group < eps_r
    GRIPPER  → max |Δgripper| < eps_g (frame 0 idle by convention)
    JOINT    → max |Δjoint|   < joint_threshold (frame 0 idle)
    RESERVED → ignored

An :class:`ActionSpec` is just ``names`` + ``types`` + ``rotation_format``.
Build one declaratively via :func:`build_action_spec` from DSL components::

    build_action_spec(Pos(), Rot("rot6d"), Gripper())             # 10D single arm
    build_action_spec(Pos(), Rot("rot6d"))                        # 9D no gripper
    build_action_spec(Joint(n=14, label="arm"),                   # 30D joint-space
                      Joint(n=14, label="end"),
                      Joint(n=2,  label="gripper"))
    build_action_spec(Pos(prefix="left"),  Rot("rot6d", "left"),  Gripper(prefix="left"),
                      Pos(prefix="right"), Rot("rot6d", "right"), Gripper(prefix="right"))

Naming convention:
    Default ``pos_x``, ``rot_0``, ``gripper``, ``arm_0`` ...
    With ``prefix="left"`` (idempotent on trailing ``_``): ``left_pos_x`` ...
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from cosmos_framework.data.generator.action.utils.pose_utils import (
    RotationConvention,
    _identity_rotation_vector,
)


class DimType(str, Enum):
    """Per-column action-dim category (drives idle detection)."""

    POS = "pos"
    ROT = "rot"
    GRIPPER = "gripper"
    JOINT = "joint"
    RESERVED = "reserved"


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Structural description of an action vector: names + per-dim types.

    All ROT dims share a single ``rotation_format``; mixed formats in one spec
    are not supported (raise at build time).

    This struct contains no detection thresholds — those are passed at call
    time to :func:`compute_idle_frames` so each dataset can tune them
    independently of layout.
    """

    names: list[str]
    types: list[DimType]
    rotation_format: RotationConvention = "rot6d"

    @property
    def dim(self) -> int:
        return len(self.names)


# ---------------------------------------------------------------------------
# DSL components
# ---------------------------------------------------------------------------


def _join_prefix(prefix: str, name: str) -> str:
    """Join ``prefix`` and ``name`` with a single ``_``; idempotent on trailing ``_``."""
    return name if not prefix else f"{prefix.rstrip('_')}_{name}"


@dataclass(frozen=True)
class Pos:
    """Translation block.

    Default 3D (``pos_x``, ``pos_y``, ``pos_z``). For planar tasks (e.g. PushT)
    use ``Pos(dim=2)`` → ``pos_x``, ``pos_y``. ``dim >= 4`` falls back to
    indexed names ``pos_0``, ``pos_1``, ...
    """

    dim: int = 3
    prefix: str = ""
    type: ClassVar[DimType] = DimType.POS

    def names(self) -> list[str]:
        if self.dim <= 3:
            return [_join_prefix(self.prefix, f"pos_{c}") for c in "xyz"[: self.dim]]
        return [_join_prefix(self.prefix, f"pos_{i}") for i in range(self.dim)]


@dataclass(frozen=True)
class Rot:
    """Rotation block; ``format`` selects the encoding.

    Supported formats and per-dim names:

    - ``rot6d``      → 6 dims, ``rot_0`` ... ``rot_5``     (identity ``[1,0,0,0,1,0]``)
    - ``rot9d``      → 9 dims, ``rot_0`` ... ``rot_8``     (identity ``[1,0,0,0,1,0,0,0,1]``)
    - ``euler_xyz``  → 3 dims, ``roll``, ``pitch``, ``yaw`` (identity ``[0,0,0]``)
    - ``axisangle``  → 3 dims, ``axang_x/y/z``              (identity ``[0,0,0]``)
    - ``quat_xyzw`` / ``quat_wxyz`` → 4 dims, ``quat_x/y/z/w`` in declared order
    """

    format: RotationConvention = "rot6d"
    prefix: str = ""
    type: ClassVar[DimType] = DimType.ROT

    @property
    def rotation_format(self) -> RotationConvention:
        return self.format

    @property
    def dim(self) -> int:
        return _identity_rotation_vector(self.format).shape[0]

    def names(self) -> list[str]:
        if self.format == "euler_xyz":
            return [_join_prefix(self.prefix, c) for c in ("roll", "pitch", "yaw")]
        if self.format == "axisangle":
            return [_join_prefix(self.prefix, f"axang_{c}") for c in "xyz"]
        if self.format.startswith("quat_"):
            order = self.format.split("_", 1)[1]  # "xyzw" or "wxyz"
            return [_join_prefix(self.prefix, f"quat_{c}") for c in order]
        return [_join_prefix(self.prefix, f"rot_{i}") for i in range(self.dim)]


@dataclass(frozen=True)
class Gripper:
    """1D gripper command (binary 0/1 or continuous). Detected by frame-diff."""

    prefix: str = ""
    type: ClassVar[DimType] = DimType.GRIPPER

    @property
    def dim(self) -> int:
        return 1

    def names(self) -> list[str]:
        return [_join_prefix(self.prefix, "gripper")]


@dataclass(frozen=True)
class Joint:
    """``n`` joint commands. Detected by frame-diff against ``joint_threshold``."""

    n: int = 0
    label: str = "joint"
    prefix: str = ""
    type: ClassVar[DimType] = DimType.JOINT

    @property
    def dim(self) -> int:
        return self.n

    def names(self) -> list[str]:
        return [_join_prefix(self.prefix, f"{self.label}_{i}") for i in range(self.n)]


@dataclass(frozen=True)
class Reserved:
    """``n`` dims counted in ``action_dim`` but ignored by idle detection."""

    n: int = 0
    label: str = "reserved"
    prefix: str = ""
    type: ClassVar[DimType] = DimType.RESERVED

    @property
    def dim(self) -> int:
        return self.n

    def names(self) -> list[str]:
        return [_join_prefix(self.prefix, f"{self.label}_{i}") for i in range(self.n)]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


# Type alias for any DSL component. Not a runtime check — only annotation hint.
Component = Pos | Rot | Gripper | Joint | Reserved


def build_action_spec(*components: Component) -> ActionSpec:
    """Compose ``components`` into an :class:`ActionSpec`.

    Each component contributes its ``names()`` and replicates its ``type`` for
    every column it occupies. The first ROT component's ``rotation_format``
    is captured for the whole spec; mixing formats raises ``ValueError``.
    """
    names: list[str] = []
    types: list[DimType] = []
    rotation_format: RotationConvention | None = None

    for c in components:
        names.extend(c.names())
        types.extend([c.type] * c.dim)
        if c.type == DimType.ROT:
            fmt = c.rotation_format  # type: ignore[union-attr]
            if rotation_format is None:
                rotation_format = fmt
            elif rotation_format != fmt:
                raise ValueError(f"Mixed rotation_format in one ActionSpec: {rotation_format!r} vs {fmt!r}")

    return ActionSpec(
        names=names,
        types=types,
        rotation_format=rotation_format or "rot6d",
    )


__all__ = [
    "ActionSpec",
    "Component",
    "DimType",
    "Gripper",
    "Joint",
    "Pos",
    "Reserved",
    "Rot",
    "build_action_spec",
]

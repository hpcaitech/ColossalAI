# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared AgiBot metadata used by datasets and visualizers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

AgibotWorldKind = Literal["gripper"]

AGIBOT_WORLD_URDF_FILENAME = "G1_omnipicker_calibrated.urdf"
AGIBOT_WORLD_ARM_STATE_SLICE = slice(0, 14)
AGIBOT_WORLD_STATE_HEAD_YAW_IDX = 16
AGIBOT_WORLD_STATE_HEAD_PITCH_IDX = 17
AGIBOT_WORLD_STATE_WAIST_PITCH_IDX = 18
AGIBOT_WORLD_STATE_WAIST_LIFT_IDX = 19
AGIBOT_WORLD_HEAD_PITCH_JOINT_NAME = "idx04_head_pitch_joint"

# -- Ext layout constants (94-dim state) -------------------------------------
# The ext split stores joints at different offsets from the standard layout.
AGIBOT_WORLD_EXT_ARM_STATE_SLICE = slice(54, 68)
AGIBOT_WORLD_EXT_STATE_HEAD_YAW_IDX = 82
AGIBOT_WORLD_EXT_STATE_HEAD_PITCH_IDX = 83
AGIBOT_WORLD_EXT_STATE_WAIST_PITCH_IDX = 84
AGIBOT_WORLD_EXT_STATE_WAIST_LIFT_IDX = 85
AGIBOT_WORLD_EXT_STATE_ROBOT_POSITION_SLICE = slice(86, 89)
AGIBOT_WORLD_EXT_STATE_ROBOT_ORIENTATION_SLICE = slice(89, 93)
AGIBOT_WORLD_EXT_STATE_LEFT_HAND_SLICE = slice(0, 1)
AGIBOT_WORLD_EXT_STATE_RIGHT_HAND_SLICE = slice(1, 2)
AGIBOT_WORLD_HEAD_CAMERA_LINK_NAME = "head_camera_link"
AGIBOT_WORLD_LEFT_EE_LINK_NAME = "gripper_l_base_link"
AGIBOT_WORLD_RIGHT_EE_LINK_NAME = "gripper_r_base_link"
AGIBOT_WORLD_ARM_JOINT_NAMES_LEFT = tuple(f"idx{4 + i:02d}_left_arm_joint{i}" for i in range(1, 8))
AGIBOT_WORLD_ARM_JOINT_NAMES_RIGHT = tuple(f"idx{11 + i:02d}_right_arm_joint{i}" for i in range(1, 8))
AGIBOT_WORLD_WAIST_LIFT_JOINT_NAME = "idx01_waist_lift_joint"
AGIBOT_WORLD_WAIST_PITCH_JOINT_NAME = "idx02_waist_pitch_joint"
AGIBOT_WORLD_HEAD_YAW_JOINT_NAME = "idx03_head_yaw_joint"
AGIBOT_WORLD_GRIPPER_OPEN_ANGLE_RAD = math.pi / 4.0
AGIBOT_WORLD_GRIPPER_OPEN_ACTUATOR_DEG = 120.0
AGIBOT_WORLD_LEFT_GRIPPER_JOINT_MIMICS = (
    ("idx31_gripper_l_inner_joint1", 1.0, 0.0),
    ("idx32_gripper_l_inner_joint3", 0.1, 0.0),
    ("idx33_gripper_l_inner_joint4", 0.25, 0.0),
    ("idx39_gripper_l_inner_joint0", -0.7, 0.0),
    ("idx41_gripper_l_outer_joint1", -1.0, 0.0),
    ("idx42_gripper_l_outer_joint3", 0.1, 0.0),
    ("idx43_gripper_l_outer_joint4", -0.25, 0.0),
    ("idx49_gripper_l_outer_joint0", 0.7, 0.0),
)
AGIBOT_WORLD_RIGHT_GRIPPER_JOINT_MIMICS = (
    ("idx71_gripper_r_inner_joint1", 1.0, 0.0),
    ("idx72_gripper_r_inner_joint3", 0.1, 0.0),
    ("idx73_gripper_r_inner_joint4", 0.25, 0.0),
    ("idx79_gripper_r_inner_joint0", -0.7, 0.0),
    ("idx81_gripper_r_outer_joint1", -1.0, 0.0),
    ("idx82_gripper_r_outer_joint3", 0.1, 0.0),
    ("idx83_gripper_r_outer_joint4", -0.25, 0.0),
    ("idx89_gripper_r_outer_joint0", 0.7, 0.0),
)


@dataclass(frozen=True)
class AgibotWorldKindSpec:
    """Layout metadata shared across all embodiments of one hand kind."""

    kind: AgibotWorldKind
    state_hand_slice: slice


@dataclass(frozen=True)
class AgibotWorldEmbodimentSpec:
    """Per-embodiment metadata shared by training and visualization code."""

    embodiment_type: str
    kind: AgibotWorldKind


AGIBOT_WORLD_KIND_SPECS: dict[AgibotWorldKind, AgibotWorldKindSpec] = {
    "gripper": AgibotWorldKindSpec(
        kind="gripper",
        state_hand_slice=slice(14, 16),
    ),
}

AGIBOT_WORLD_EMBODIMENT_SPECS: dict[str, AgibotWorldEmbodimentSpec] = {
    "agibot_world_gripper": AgibotWorldEmbodimentSpec(
        embodiment_type="agibot_world_gripper",
        kind="gripper",
    ),
    "agibot_world_gripper_ext": AgibotWorldEmbodimentSpec(
        embodiment_type="agibot_world_gripper_ext",
        kind="gripper",
    ),
}


def get_agibot_world_embodiment_spec(embodiment_type: str) -> AgibotWorldEmbodimentSpec:
    """Return the registered spec for one AgiBot embodiment."""

    try:
        return AGIBOT_WORLD_EMBODIMENT_SPECS[embodiment_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown AgiBot World embodiment_type={embodiment_type!r}. "
            f"Expected one of {sorted(AGIBOT_WORLD_EMBODIMENT_SPECS)}."
        ) from exc


def get_agibot_world_kind_spec(embodiment_type: str | AgibotWorldKind) -> AgibotWorldKindSpec:
    """Resolve an embodiment type or kind to its shared layout metadata."""

    kind = embodiment_type if embodiment_type in AGIBOT_WORLD_KIND_SPECS else get_agibot_world_kind(embodiment_type)
    return AGIBOT_WORLD_KIND_SPECS[kind]


def get_agibot_world_kind(embodiment_type: str) -> AgibotWorldKind:
    """Return the hand kind used by one AgiBot embodiment."""

    return get_agibot_world_embodiment_spec(embodiment_type).kind


def get_agibot_world_urdf_path() -> Path:
    """Return the committed AgiBot G1 omnipicker URDF path."""

    return Path(__file__).resolve().parent / "robot_assets" / AGIBOT_WORLD_URDF_FILENAME

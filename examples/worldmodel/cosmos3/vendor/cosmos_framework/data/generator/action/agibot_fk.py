# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Lightweight AgiBot World forward kinematics for datasets and viewers."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np

from cosmos_framework.data.generator.action.agibot_spec import (
    AGIBOT_WORLD_ARM_JOINT_NAMES_LEFT,
    AGIBOT_WORLD_ARM_JOINT_NAMES_RIGHT,
    AGIBOT_WORLD_ARM_STATE_SLICE,
    AGIBOT_WORLD_EXT_ARM_STATE_SLICE,
    AGIBOT_WORLD_EXT_STATE_HEAD_PITCH_IDX,
    AGIBOT_WORLD_EXT_STATE_HEAD_YAW_IDX,
    AGIBOT_WORLD_EXT_STATE_LEFT_HAND_SLICE,
    AGIBOT_WORLD_EXT_STATE_RIGHT_HAND_SLICE,
    AGIBOT_WORLD_EXT_STATE_ROBOT_ORIENTATION_SLICE,
    AGIBOT_WORLD_EXT_STATE_ROBOT_POSITION_SLICE,
    AGIBOT_WORLD_EXT_STATE_WAIST_LIFT_IDX,
    AGIBOT_WORLD_EXT_STATE_WAIST_PITCH_IDX,
    AGIBOT_WORLD_GRIPPER_OPEN_ACTUATOR_DEG,
    AGIBOT_WORLD_GRIPPER_OPEN_ANGLE_RAD,
    AGIBOT_WORLD_HEAD_CAMERA_LINK_NAME,
    AGIBOT_WORLD_HEAD_PITCH_JOINT_NAME,
    AGIBOT_WORLD_HEAD_YAW_JOINT_NAME,
    AGIBOT_WORLD_LEFT_EE_LINK_NAME,
    AGIBOT_WORLD_LEFT_GRIPPER_JOINT_MIMICS,
    AGIBOT_WORLD_RIGHT_EE_LINK_NAME,
    AGIBOT_WORLD_RIGHT_GRIPPER_JOINT_MIMICS,
    AGIBOT_WORLD_STATE_HEAD_PITCH_IDX,
    AGIBOT_WORLD_STATE_HEAD_YAW_IDX,
    AGIBOT_WORLD_STATE_WAIST_LIFT_IDX,
    AGIBOT_WORLD_STATE_WAIST_PITCH_IDX,
    AGIBOT_WORLD_WAIST_LIFT_JOINT_NAME,
    AGIBOT_WORLD_WAIST_PITCH_JOINT_NAME,
    get_agibot_world_embodiment_spec,
    get_agibot_world_kind_spec,
    get_agibot_world_urdf_path,
)
from cosmos_framework.data.generator.action.utils.pose_utils import convert_rotation

_GRIPPER_VALUE_EPS = 1e-4
_QUATERNION_NORM_EPS = 1e-8
_GRIPPER_ACTUATOR_OVERSHOOT_DEG = 5.0
# Main-branch wrist rotations composed with one extra local-Z 180 degree rotation.
AGIBOT_WORLD_LEFT_GRIPPER_TO_OPENCV: np.ndarray = np.asarray(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
AGIBOT_WORLD_RIGHT_GRIPPER_TO_OPENCV: np.ndarray = np.asarray(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
AGIBOT_WORLD_GRIPPER_TO_OPENCV_BY_WRIST: dict[str, np.ndarray] = {
    "right_wrist": AGIBOT_WORLD_RIGHT_GRIPPER_TO_OPENCV,
    "left_wrist": AGIBOT_WORLD_LEFT_GRIPPER_TO_OPENCV,
}


def _scale_to_unit_interval(values: np.ndarray, scale: float) -> np.ndarray:
    """Scale non-negative gripper actuator values to ``[0,1]``."""

    return np.clip(values / scale, 0.0, 1.0).astype(np.float32, copy=False)


def _scale_negative_to_unit_interval(values: np.ndarray, scale: float) -> np.ndarray:
    """Scale URDF-style negative gripper angles to ``[0,1]`` open fractions."""

    return np.clip(-values / scale, 0.0, 1.0).astype(np.float32, copy=False)


def _normalize_quaternions_xyzw(quaternions: np.ndarray) -> np.ndarray:
    """Normalize ``xyzw`` quaternions, treating all-zero rows as identity."""

    normalized = np.asarray(quaternions, dtype=np.float32).copy()  # [T,4]
    norms = np.linalg.norm(normalized, axis=-1, keepdims=True)  # [T,1]
    valid = norms[:, 0] >= _QUATERNION_NORM_EPS  # [T]
    normalized[valid] = normalized[valid] / norms[valid]  # [T_valid,4]
    normalized[~valid] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # [T_invalid,4]
    return normalized


def _quat_xyzw_to_rotation_matrix(quaternions: np.ndarray) -> np.ndarray:
    """Convert ``xyzw`` quaternions to rotation matrices."""

    normalized = _normalize_quaternions_xyzw(quaternions)  # [T,4]
    rotations = convert_rotation(
        normalized,
        input_format="quat_xyzw",
        output_format="matrix",
        normalize_matrix=True,
    )
    return np.asarray(rotations, dtype=np.float32)


def build_robot_base_transforms(positions: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
    """Build robot-base poses from position and ``xyzw`` quaternion arrays."""

    positions = np.asarray(positions, dtype=np.float32)  # [T,3]
    quaternions = np.asarray(quaternions, dtype=np.float32)  # [T,4]
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"robot base positions must have shape [T,3], got {positions.shape}.")
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"robot base quaternions must have shape [T,4], got {quaternions.shape}.")
    if positions.shape[0] != quaternions.shape[0]:
        raise ValueError(
            f"robot base positions/quaternions must share T, got {positions.shape[0]} and {quaternions.shape[0]}."
        )

    transforms = np.tile(np.eye(4, dtype=np.float32), (positions.shape[0], 1, 1))  # [T,4,4]
    transforms[:, :3, :3] = _quat_xyzw_to_rotation_matrix(quaternions)  # [T,3,3]
    transforms[:, :3, 3] = positions  # [T,3]
    return transforms


def _invert_rigid_transform(transform: np.ndarray) -> np.ndarray:
    """Invert one homogeneous rigid transform."""

    inverse = np.eye(4, dtype=np.float32)  # [4,4]
    rotation_t = transform[:3, :3].T.astype(np.float32, copy=False)  # [3,3]
    inverse[:3, :3] = rotation_t
    inverse[:3, 3] = -(rotation_t @ transform[:3, 3])  # [3]
    return inverse


def apply_robot_base_motion_to_poses(
    poses_by_name: dict[str, np.ndarray],
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> dict[str, np.ndarray]:
    """Apply mobile-base motion to FK poses, normalized to the first frame."""

    base_poses = build_robot_base_transforms(positions, quaternions)  # [T,4,4]
    initial_base_inv = _invert_rigid_transform(base_poses[0])  # [4,4]
    base_motion = np.einsum("ij,tjk->tik", initial_base_inv, base_poses).astype(np.float32, copy=False)  # [T,4,4]
    return {
        name: np.einsum("tij,tjk->tik", base_motion, poses).astype(np.float32, copy=False)  # [T,4,4]
        for name, poses in poses_by_name.items()
    }


def _apply_ext_base_motion_to_poses(
    poses_by_name: dict[str, np.ndarray],
    states: np.ndarray,
    embodiment_type: str,
) -> dict[str, np.ndarray]:
    """Apply ext mobile-base motion to FK poses, normalized to the first frame."""

    if embodiment_type != "agibot_world_gripper_ext":
        return poses_by_name
    if states.shape[1] < AGIBOT_WORLD_EXT_STATE_ROBOT_ORIENTATION_SLICE.stop:
        raise ValueError(
            f"agibot_world_gripper_ext state must include robot pose through index "
            f"{AGIBOT_WORLD_EXT_STATE_ROBOT_ORIENTATION_SLICE.stop - 1}, got shape {states.shape}."
        )

    positions = states[:, AGIBOT_WORLD_EXT_STATE_ROBOT_POSITION_SLICE].astype(np.float32, copy=False)  # [T,3]
    quaternions = states[:, AGIBOT_WORLD_EXT_STATE_ROBOT_ORIENTATION_SLICE].astype(np.float32, copy=False)  # [T,4]
    return apply_robot_base_motion_to_poses(poses_by_name, positions, quaternions)


def apply_agibot_gripper_to_opencv(
    poses_by_name: dict[str, np.ndarray],
    to_opencv_by_wrist: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Post-rotate AgiBot gripper wrist poses into OpenCV convention."""

    aligned = {name: poses.astype(np.float32, copy=True) for name, poses in poses_by_name.items()}  # {name:[...,4,4]}
    for wrist_name, wrist_to_opencv in to_opencv_by_wrist.items():
        poses = aligned.get(wrist_name)
        if poses is None:
            continue
        aligned[wrist_name][..., :3, :3] = poses[..., :3, :3] @ wrist_to_opencv.astype(poses.dtype)  # [...,3,3]
    return aligned


def _get_agibot_world_mujoco_kinematics_xml() -> str:
    """Build a MuJoCo-loadable kinematics-only XML string from the committed URDF."""

    root = ET.parse(get_agibot_world_urdf_path()).getroot()
    mujoco_element = root.find("mujoco")
    if mujoco_element is None:
        mujoco_element = ET.Element("mujoco")
        root.insert(0, mujoco_element)
    compiler_element = mujoco_element.find("compiler")
    if compiler_element is None:
        compiler_element = ET.SubElement(mujoco_element, "compiler")
    compiler_element.attrib["fusestatic"] = "false"

    for link_element in root.findall("link"):
        for child_element in list(link_element):
            if child_element.tag in {"visual", "collision"}:
                link_element.remove(child_element)

    return ET.tostring(root, encoding="unicode")


class _MujocoFk:
    """MuJoCo-backed FK engine for the committed AgiBot G1 omnipicker URDF."""

    def __init__(self) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(_get_agibot_world_mujoco_kinematics_xml())
        self.data = mujoco.MjData(self.model)
        self._joint_qpos_addresses: dict[str, int] = {}
        for joint_id in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name is not None:
                self._joint_qpos_addresses[joint_name] = int(self.model.jnt_qposadr[joint_id])

    def link_poses(self, joint_values: dict[str, float]) -> dict[str, np.ndarray]:
        """Return world transforms for every named body in the MuJoCo model."""

        self.data.qpos[:] = 0.0
        for joint_name, joint_value in joint_values.items():
            qpos_address = self._joint_qpos_addresses.get(joint_name)
            if qpos_address is not None:
                self.data.qpos[qpos_address] = float(joint_value)
        self._mujoco.mj_forward(self.model, self.data)

        poses: dict[str, np.ndarray] = {}
        for body_id in range(1, self.model.nbody):
            body_name = self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name is None:
                continue
            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3] = self.data.xmat[body_id].reshape(3, 3).astype(np.float32, copy=False)
            transform[:3, 3] = self.data.xpos[body_id].astype(np.float32, copy=False)
            poses[body_name] = transform
        return poses


@lru_cache(maxsize=1)
def _get_fk_engine() -> _MujocoFk:
    """Return a cached MuJoCo FK engine for the committed AgiBot URDF."""

    return _MujocoFk()


def _extract_joint_values_from_state(state: np.ndarray, embodiment_type: str) -> dict[str, float]:
    """Map one observation.state vector to the URDF joint names used for FK."""

    if embodiment_type == "agibot_world_gripper_ext":
        # Ext layout: 94-dim state with joints at different offsets.
        arm_state = state[AGIBOT_WORLD_EXT_ARM_STATE_SLICE]
        head_yaw = float(state[AGIBOT_WORLD_EXT_STATE_HEAD_YAW_IDX])
        head_pitch = float(state[AGIBOT_WORLD_EXT_STATE_HEAD_PITCH_IDX])
        waist_lift = float(state[AGIBOT_WORLD_EXT_STATE_WAIST_LIFT_IDX])
        waist_pitch = float(state[AGIBOT_WORLD_EXT_STATE_WAIST_PITCH_IDX])
    else:
        arm_state = state[AGIBOT_WORLD_ARM_STATE_SLICE]
        head_yaw = float(state[AGIBOT_WORLD_STATE_HEAD_YAW_IDX])
        head_pitch = float(state[AGIBOT_WORLD_STATE_HEAD_PITCH_IDX])
        waist_pitch = float(state[AGIBOT_WORLD_STATE_WAIST_PITCH_IDX])
        waist_lift = float(state[AGIBOT_WORLD_STATE_WAIST_LIFT_IDX])

    joint_values = {
        AGIBOT_WORLD_WAIST_LIFT_JOINT_NAME: float(waist_lift),
        AGIBOT_WORLD_WAIST_PITCH_JOINT_NAME: float(waist_pitch),
        AGIBOT_WORLD_HEAD_YAW_JOINT_NAME: float(head_yaw),
        AGIBOT_WORLD_HEAD_PITCH_JOINT_NAME: float(head_pitch),
    }
    joint_values.update({name: float(arm_state[idx]) for idx, name in enumerate(AGIBOT_WORLD_ARM_JOINT_NAMES_LEFT)})
    joint_values.update({name: float(arm_state[7 + idx]) for idx, name in enumerate(AGIBOT_WORLD_ARM_JOINT_NAMES_RIGHT)})
    _set_gripper_joint_values_from_state(joint_values, state, embodiment_type)
    return joint_values


def _set_gripper_joint_values_from_state(
    joint_values: dict[str, float],
    state: np.ndarray,
    embodiment_type: str,
) -> None:
    """Map observed scalar gripper state into all omnipicker finger joints."""

    embodiment_spec = get_agibot_world_embodiment_spec(embodiment_type)
    if embodiment_spec.kind != "gripper":
        return

    if embodiment_type == "agibot_world_gripper_ext":
        left_raw = float(state[AGIBOT_WORLD_EXT_STATE_LEFT_HAND_SLICE][0])
        right_raw = float(state[AGIBOT_WORLD_EXT_STATE_RIGHT_HAND_SLICE][0])
    else:
        kind_spec = get_agibot_world_kind_spec(embodiment_type)
        state_hand_slice = kind_spec.state_hand_slice
        left_raw = float(state[state_hand_slice.start])
        right_raw = float(state[state_hand_slice.start + 1])

    left_open = float(convert_gripper_state_to_open_fraction(np.asarray([left_raw], dtype=np.float32))[0])  # [1]
    right_open = float(convert_gripper_state_to_open_fraction(np.asarray([right_raw], dtype=np.float32))[0])  # [1]
    for opening, joint_mimics in (
        (left_open, AGIBOT_WORLD_LEFT_GRIPPER_JOINT_MIMICS),
        (right_open, AGIBOT_WORLD_RIGHT_GRIPPER_JOINT_MIMICS),
    ):
        primary_angle = -float(np.clip(opening, 0.0, 1.0)) * AGIBOT_WORLD_GRIPPER_OPEN_ANGLE_RAD
        for joint_name, multiplier, offset in joint_mimics:
            joint_values[joint_name] = multiplier * primary_angle + offset


def compute_fk_transforms(
    state: np.ndarray,
    embodiment_type: str,
) -> dict[str, np.ndarray]:
    """Compute native-frame calibrated head-camera and gripper-base transforms for one state."""

    fk_engine = _get_fk_engine()
    link_poses = fk_engine.link_poses(_extract_joint_values_from_state(state, embodiment_type))

    return {
        "head_camera": link_poses[AGIBOT_WORLD_HEAD_CAMERA_LINK_NAME].astype(np.float32, copy=False),
        "right_wrist": link_poses[AGIBOT_WORLD_RIGHT_EE_LINK_NAME].astype(np.float32, copy=False),
        "left_wrist": link_poses[AGIBOT_WORLD_LEFT_EE_LINK_NAME].astype(np.float32, copy=False),
    }


def compute_fk_transforms_batch(
    states: np.ndarray,
    embodiment_type: str,
) -> dict[str, np.ndarray]:
    """Compute absolute transforms for a batch of AgiBot observation states."""

    num_steps = int(states.shape[0])
    head_camera = np.empty((num_steps, 4, 4), dtype=np.float32)
    right_wrist = np.empty((num_steps, 4, 4), dtype=np.float32)
    left_wrist = np.empty((num_steps, 4, 4), dtype=np.float32)

    for step in range(num_steps):
        transforms = compute_fk_transforms(states[step], embodiment_type)
        head_camera[step] = transforms["head_camera"]
        right_wrist[step] = transforms["right_wrist"]
        left_wrist[step] = transforms["left_wrist"]

    transforms_by_name = {
        "head_camera": head_camera,
        "right_wrist": right_wrist,
        "left_wrist": left_wrist,
    }
    return _apply_ext_base_motion_to_poses(transforms_by_name, states, embodiment_type)


def convert_gripper_state_to_open_fraction(values: np.ndarray) -> np.ndarray:
    """Convert observed AgiBot gripper state to viewer/dataset open fractions.

    The shared viewer/action convention is ``0=closed`` and ``1=open``.
    Observed AgiBot gripper state uses actuator-close angle units: ``0`` is
    open and ``120`` is closed. Some episodes contain small closed-state
    overshoot above ``120``; those values are accepted and clipped to fully
    closed. Small open-state sensor jitter such as ``0.217`` must therefore
    remain nearly fully open, not be interpreted as a normalized close fraction.
    """

    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    if not np.isfinite(values).all():
        raise ValueError("AgiBot gripper values contain NaN or Inf values.")

    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if (
        min_value < -_GRIPPER_VALUE_EPS
        and min_value >= -AGIBOT_WORLD_GRIPPER_OPEN_ANGLE_RAD - _GRIPPER_VALUE_EPS
        and max_value <= _GRIPPER_VALUE_EPS
    ):
        return _scale_negative_to_unit_interval(values, AGIBOT_WORLD_GRIPPER_OPEN_ANGLE_RAD)
    if (
        min_value < -_GRIPPER_VALUE_EPS
        and min_value >= -np.degrees(AGIBOT_WORLD_GRIPPER_OPEN_ANGLE_RAD) - _GRIPPER_VALUE_EPS
        and max_value <= _GRIPPER_VALUE_EPS
    ):
        return _scale_negative_to_unit_interval(values, np.degrees(AGIBOT_WORLD_GRIPPER_OPEN_ANGLE_RAD))
    max_actuator_value = AGIBOT_WORLD_GRIPPER_OPEN_ACTUATOR_DEG + _GRIPPER_ACTUATOR_OVERSHOOT_DEG
    if min_value >= -_GRIPPER_VALUE_EPS and max_value <= max_actuator_value + _GRIPPER_VALUE_EPS:
        close_fraction = _scale_to_unit_interval(values, AGIBOT_WORLD_GRIPPER_OPEN_ACTUATOR_DEG)  # [*]
        return (1.0 - close_fraction).astype(np.float32, copy=False)  # [*]

    raise ValueError(
        f"Unsupported AgiBot gripper value range; min={min_value:.4f}, max={max_value:.4f}. "
        f"Expected URDF angle [-pi/4,0] or actuator-close degrees [0,{max_actuator_value:.1f}] "
        f"(values above {AGIBOT_WORLD_GRIPPER_OPEN_ACTUATOR_DEG:.1f} are clipped closed)."
    )



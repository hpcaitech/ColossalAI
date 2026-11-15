# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Small LIBERO pose helpers shared by training and closed-loop eval."""

from __future__ import annotations

import numpy as np
import torch

from cosmos_framework.data.generator.action.utils.pose_utils import (
    RotationConvention,
    build_abs_pose_from_components,
)

# Local-frame post-rotation pattern:
# R_opencv = R_native @ *_TO_OPENCV.
LIBERO_TO_OPENCV: np.ndarray = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

LIBERO_ROTATION_FORMATS: dict[str, RotationConvention] = {
    "3d": "axisangle",
    "6d": "rot6d",
    "9d": "rot9d",
}
LIBERO_ACTION_DIMS: dict[str, int] = {"3d": 7, "6d": 10, "9d": 13}


def libero_rotation_format(rotation_space: str) -> RotationConvention:
    """Return the shared ``pose_utils`` rotation format for a LIBERO setting."""
    rotation_format = LIBERO_ROTATION_FORMATS.get(rotation_space)
    if rotation_format is None:
        raise ValueError(f"Unsupported rotation_space={rotation_space!r}. Use 3d/6d/9d.")
    return rotation_format


def libero_action_dim(rotation_space: str) -> int:
    """Return ``[xyz, rotation, gripper]`` action width for LIBERO."""
    action_dim = LIBERO_ACTION_DIMS.get(rotation_space)
    if action_dim is None:
        raise ValueError(f"Unsupported rotation_space={rotation_space!r}. Use 3d/6d/9d.")
    return action_dim


def libero_rotation_space_from_action_dim(action_dim: int) -> str:
    """Infer LIBERO rotation space from unpadded action width."""
    for rotation_space, dim in LIBERO_ACTION_DIMS.items():
        if dim == action_dim:
            return rotation_space
    raise ValueError(f"Unable to infer rotation_space from action_dim={action_dim}.")


def build_libero_abs_pose(state_raw: torch.Tensor | np.ndarray, *, to_opencv: bool) -> np.ndarray:
    """Build absolute LIBERO EE poses from state rows.

    ``state_raw`` is ``[x,y,z,axisangle(3),gripper(2)]``.  When requested, the
    local EE frame is post-rotated into the shared OpenCV-style action frame.
    """
    if isinstance(state_raw, torch.Tensor):
        state_np = state_raw.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        state_np = np.asarray(state_raw, dtype=np.float32)

    poses_abs = build_abs_pose_from_components(state_np[:, :3], state_np[:, 3:6], "axisangle")
    if to_opencv:
        poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ LIBERO_TO_OPENCV
    return poses_abs

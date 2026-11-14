# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fractal LeRobot dataset — Google Robot RT-1.

Robot: google_robot
87,212 episodes, 3,786,400 frames, 599 tasks, fps=3
state: [x, y, z, rx, ry, rz, rw, gripper]  (8D, quaternion)
action: [x, y, z, roll, pitch, yaw, gripper] (7D, delta)
video:  observation.images.image (256×320)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from cosmos_framework.data.generator.action.utils.action_spec import ActionSpec, Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.generator.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.generator.action.utils.pose_utils import (
    build_abs_pose_from_components,
    pose_abs_to_rel,
)

PoseConvention = Literal["backward_framewise"]
Viewpoint = Literal["ego_view"]

_IMAGE_FEATURE = "observation.images.image"
_STATE_FEATURE = "observation.state"
_ACTION_FEATURE = "action"

# These episodes contain base motion, which breaks the fixed-base Google Robot
# action assumption used by training and the viewer.
_SKIPPED_EPISODE_IDS: frozenset[int] = frozenset({29, 189, 382})

# Google Robot raw EE frame has x/y axes rotated ~90° around z compared to
# OpenCV convention.  Rz(-90°) as a right-multiply corrects this:
#   new_x = -old_y (rightward), new_y = old_x (downward), z unchanged (approach).
_GOOGLE_ROBOT_TO_OPENCV = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)

# ---------------------------------------------------------------------------
# TCP → flange (gripper body) offset
# ---------------------------------------------------------------------------
# The fractal dataset records EE poses at ``link_gripper_tcp`` — a calibrated
# tool-center-point 164 mm past the gripper body (``link_gripper``), roughly
# at the fingertip.  Re-referencing to ``link_gripper`` removes the
# calibration-dependent tilt and places the frame at the last actuated link.
#
# T = oMf[link_gripper_tcp]⁻¹ · oMf[link_gripper], computed via pinocchio FK
# at the neutral config from the SimplerEnv URDF.
# fmt: off
_TCP_TO_FLANGE = np.array([
    [+0.9999897671, -0.0008686425, +0.0044397163, -0.0050618476],
    [+0.0008745501, +0.9999987346, -0.0013288658, -0.0016717725],
    [-0.0044385564, +0.0013327349, +0.9999892615, -0.1635144743],
    [+0.0000000000, +0.0000000000, +0.0000000000, +1.0000000000],
], dtype=np.float32)
# fmt: on

_NORMALIZER_PATH = Path(__file__).parent.parent / "normalizer_stats/fractal_lerobot_stats.json"


class FractalLeRobotDataset(ActionBaseDataset):
    """Fractal (Google RT-1) dataset with 10D cartesian actions:

        [pos_delta(3), rot6d_delta(6), gripper(1)]

    Expects a LeRobot v2 dataset with:
      * ``observation.images.image``: ego-view RGB video (256×320).
      * ``observation.state``: 8D EE pose ``[x, y, z, rx, ry, rz, rw, gripper]``
        in TCP frame with quaternion (x, y, z, w) order.
      * ``action``: 7D delta ``[x, y, z, roll, pitch, yaw, gripper]``; only the
        gripper column (index 6) is used — SE(3) actions are derived from state.

    Episodes in ``_SKIPPED_EPISODE_IDS`` (base-motion outliers) are dropped.
    """

    def __init__(
        self,
        root: str,
        fps: float = 3.0,
        chunk_length: int = 16,
        mode: str = "joint",
        pose_convention: PoseConvention = "backward_framewise",
        tolerance_s: float = 1e-4,
        viewpoint: Viewpoint = "ego_view",
        action_normalization: str | None = None,
        sample_stride: int = 1,
    ) -> None:
        if viewpoint != "ego_view":
            raise NotImplementedError("FractalLeRobotDataset only supports ego_view.")
        super().__init__(
            root=root,
            domain_name="fractal",
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention=pose_convention,
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            sample_stride=sample_stride,
        )

        # Drop rows belonging to known-bad episodes (base motion outliers).
        n_before = len(self._rows)
        self._rows = [row for row in self._rows if int(row["episode_index"]) not in _SKIPPED_EPISODE_IDS]
        n_dropped = n_before - len(self._rows)
        if n_dropped:
            import logging
            logging.getLogger(__name__).info(
                "FractalLeRobotDataset: dropped %d / %d rows from episodes %s",
                n_dropped,
                n_before,
                sorted(_SKIPPED_EPISODE_IDS),
            )

    @property
    def action_dim(self) -> int:
        """Action dimensionality: position(3) + 6D rotation(6) + gripper(1) = 10."""
        return 10

    def _action_spec(self) -> ActionSpec:
        return build_action_spec(Pos(dim=3), Rot("rot6d"), Gripper())

    @classmethod
    def _stats_path(cls) -> Path:
        return _NORMALIZER_PATH

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx)
        row_idx = idx * self._sample_stride
        observation_rows = self._rows[row_idx : row_idx + self._chunk_length + 1]
        action_rows = observation_rows[: self._chunk_length]

        episode = self._episodes[int(observation_rows[0]["episode_index"])]
        task = self._tasks[int(observation_rows[0]["task_index"])]
        ai_caption = random.choice([part.strip() for part in task.split(" | ") if part.strip()] or [task])

        video = self._load_video(episode, observation_rows)
        raw_action, initial_pose = self._build_raw_action(observation_rows, action_rows)

        return self._build_result(
            mode=mode,
            video=video,
            action=raw_action,
            ai_caption=ai_caption,
            initial_pose=initial_pose,
        )

    def _load_video(self, episode: dict[str, Any], observation_rows: list[dict[str, Any]]) -> torch.Tensor:
        # lerobot is a heavy, optional ("train" extra) dependency; import lazily.
        from lerobot.datasets.video_utils import decode_video_frames

        timestamps = [float(row["timestamp"]) for row in observation_rows]
        return decode_video_frames(
            self._video_path(episode, _IMAGE_FEATURE),
            [float(episode.get(f"videos/{_IMAGE_FEATURE}/from_timestamp", 0.0)) + ts for ts in timestamps],
            self._tolerance_s,
        )

    def _build_raw_action(
        self,
        observation_rows: list[dict[str, Any]],
        action_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # State layout: [x, y, z, rx, ry, rz, rw, gripper]  (T+1 frames)
        # Quaternion order: (rx, ry, rz, rw) matches scipy's (x, y, z, w).
        state = np.asarray([row[_STATE_FEATURE] for row in observation_rows], dtype=np.float32)  # [T+1, 8]
        poses_abs = build_abs_pose_from_components(state[:, 0:3], state[:, 3:7], "quat_xyzw")  # [T+1, 4, 4]

        # 1. TCP → flange: shift from link_gripper_tcp to link_gripper
        poses_abs = poses_abs @ _TCP_TO_FLANGE
        # 2. Kinematics → OpenCV convention (rotation only)
        poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ _GOOGLE_ROBOT_TO_OPENCV

        initial_pose = torch.from_numpy(poses_abs[0].copy()).float()
        poses_rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention=self._pose_convention)

        gripper = np.asarray(
            [row[_ACTION_FEATURE][6] for row in action_rows], dtype=np.float32
        ).reshape(-1, 1)  # [T, 1]

        action = np.concatenate([poses_rel[-self._chunk_length :], gripper[-self._chunk_length :]], axis=-1)
        return torch.from_numpy(action).float(), initial_pose

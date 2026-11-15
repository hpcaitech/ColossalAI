# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""UMI LeRobot dataset."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from cosmos_framework.data.generator.action.action_normalization import load_action_stats
from cosmos_framework.data.generator.action.utils.action_spec import ActionSpec, Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.generator.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.generator.action.utils.pose_utils import (
    build_abs_pose_from_components,
    pose_abs_to_rel,
)

PoseConvention = Literal["backward_framewise"]
Viewpoint = Literal["wrist_view"]

# Feature keys matching UMI LeRobot parquet columns.
# Trajectory: 7D [pos(3), quat_wxyz(4)] — the main-camera TCP pose.
_TRAJ_KEY = "observation.state.right_main_camera_trajectory_xyz_wxyz"
_GRIPPER_KEY = "observation.state.right_gripper_width_m"
_IMAGE_FEATURE = "observation.image.right_main_camera_rgb"

# Default EEF-in-camera-frame offset (most UMI rigs).
# touch_in_the_wild / FastUMI use FORWARD_EEF_IN_CAMERA_FRAME_XYZ_WXYZ with z=0.056.
_DEFAULT_EEF_IN_CAMERA_FRAME_XYZ_WXYZ: tuple[float, ...] = (0.0, 0.086, 0.09, 1.0, 0.0, 0.0, 0.0)
FORWARD_EEF_IN_CAMERA_FRAME_XYZ_WXYZ: tuple[float, ...] = (0.0, 0.086, 0.056, 1.0, 0.0, 0.0, 0.0)
"""EEF offset for touch_in_the_wild / FastUMI rigs (camera mounted slightly forward)."""

_NORMALIZER_PATH = Path(__file__).parent.parent / "normalizer_stats/umi_lerobot_stats.json"

# Action layout: single-arm is the first 10D of the 20D bimanual stats file
# (right_eef_poses(9) + right_eef_commands(1)).
_SINGLE_ARM_ACTION_DIM = 10


class UMILeRobotDataset(ActionBaseDataset):
    """UMI dataset converted to LeRobot format with 10D cartesian actions:

        [pos_delta(3), rot6d_delta(6), gripper_width(1)]

    Expects a LeRobot v2 dataset with:
      * ``observation.images.camera0``: wrist-mounted RGB video (configurable
        via ``image_key``).
      * ``observation.state.right_main_camera_trajectory_xyz_wxyz``: 7D camera
        TCP pose ``[pos(3), quat_wxyz(4)]`` for frames [0 .. chunk_length].
      * ``observation.state.right_gripper_width_m``: scalar gripper width for
        frames [1 .. chunk_length] (commanded future widths).

    Poses are transformed from the camera TCP frame to the EEF frame via
    ``eef_in_camera_frame_xyz_wxyz``, then converted to backward-framewise
    rot6d relative poses.  The stats file stores 20D bimanual stats (right +
    left arm); single-arm normalization uses only the first 10D (right arm).
    """

    def __init__(
        self,
        root: str,
        fps: float = 10.0,
        chunk_length: int = 16,
        mode: str = "joint",
        pose_convention: PoseConvention = "backward_framewise",
        tolerance_s: float = 1e-4,
        viewpoint: Viewpoint = "wrist_view",
        action_normalization: str | None = "quantile",
        sample_stride: int = 1,
        image_key: str = _IMAGE_FEATURE,
        eef_in_camera_frame_xyz_wxyz: tuple[float, ...] = _DEFAULT_EEF_IN_CAMERA_FRAME_XYZ_WXYZ,
    ) -> None:
        if viewpoint != "wrist_view":
            raise NotImplementedError("This UMI dataset only supports wrist_view.")
        super().__init__(
            root=root,
            domain_name="umi",
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention=pose_convention,
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            sample_stride=sample_stride,
        )
        self._image_key = image_key

        xyz_wxyz = np.asarray(eef_in_camera_frame_xyz_wxyz, dtype=np.float32).reshape(1, 7)
        self._eef_in_camera_frame_mat: np.ndarray = build_abs_pose_from_components(
            xyz_wxyz[:, :3], xyz_wxyz[:, 3:], "quat_wxyz"
        )[0]  # [4, 4]

    @property
    def action_dim(self) -> int:
        return _SINGLE_ARM_ACTION_DIM

    def _action_spec(self) -> ActionSpec:
        return build_action_spec(Pos(), Rot("rot6d"), Gripper())

    @classmethod
    def _stats_path(cls) -> Path:
        return _NORMALIZER_PATH

    @classmethod
    def load_action_stats(cls) -> dict[str, torch.Tensor]:
        # Stats file stores 20D bimanual layout (right + left arm).
        # Single-arm normalization uses only the first 10D (right arm).
        raw = {
            key: torch.from_numpy(value).float()
            for key, value in load_action_stats(str(cls._stats_path())).items()
        }
        return {key: tensor[:_SINGLE_ARM_ACTION_DIM] for key, tensor in raw.items()}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx)
        row_idx = idx * self._sample_stride
        # T+1 rows: current frame + T future frames
        observation_rows = self._rows[row_idx : row_idx + self._chunk_length + 1]

        episode = self._episodes[int(observation_rows[0]["episode_index"])]
        video = self._load_video(episode, observation_rows)
        raw_action, initial_pose = self._build_raw_action(observation_rows)
        task = self._tasks[int(observation_rows[0]["task_index"])]
        ai_caption = random.choice([part.strip() for part in task.split(" | ") if part.strip()] or [task])

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
            self._video_path(episode, self._image_key),
            [float(episode.get(f"videos/{self._image_key}/from_timestamp", 0.0)) + ts for ts in timestamps],
            self._tolerance_s,
        )

    def _build_raw_action(
        self,
        observation_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Trajectory: T+1 poses, [pos(3), quat_wxyz(4)] per frame.
        traj = np.asarray([row[_TRAJ_KEY] for row in observation_rows], dtype=np.float32)  # [T+1, 7]
        poses_abs = build_abs_pose_from_components(traj[:, :3], traj[:, 3:], "quat_wxyz")  # [T+1, 4, 4]

        initial_pose = torch.from_numpy(poses_abs[0].copy()).float()

        # Transform from camera TCP frame to EEF frame, then compute relative poses.
        eef_poses_abs = poses_abs @ self._eef_in_camera_frame_mat  # [T+1, 4, 4]
        eef_poses_rel = pose_abs_to_rel(
            eef_poses_abs, rotation_format="rot6d", pose_convention=self._pose_convention
        )  # [T, 9]

        # Gripper command: future frames only (rows[1:]), matching gripper_indices=[1..T].
        gripper_rows = observation_rows[1:]
        gripper_vals = [row[_GRIPPER_KEY] for row in gripper_rows]
        gripper = np.asarray(
            [float(v) if np.isscalar(v) else float(v[0]) for v in gripper_vals],
            dtype=np.float32,
        ).reshape(-1, 1)  # [T, 1]

        action = np.concatenate([eef_poses_rel, gripper], axis=-1)  # [T, 10]
        return torch.from_numpy(action).float(), initial_pose

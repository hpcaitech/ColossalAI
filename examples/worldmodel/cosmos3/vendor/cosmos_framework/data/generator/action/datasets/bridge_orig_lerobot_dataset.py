# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Bridge Orig LeRobot dataset."""

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

_IMAGE_FEATURE = "observation.images.image_0"
_STATE_FEATURE = "observation.state"
_ACTION_FEATURE = "action"

# Raw Bridge state -> kinematics frame. The WidowX controller records
# R_state = R_fk @ DEFAULT_ROTATION.T, so R_fk = R_state @ DEFAULT_ROTATION.
_DEFAULT_ROTATION = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    dtype=np.float32,
)

# Kinematics frame -> OpenCV frame used by Cosmos action.
_BRIDGE_TO_OPENCV = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)

# Re-reference from ee_gripper_link to gripper_link in the kinematics frame.
_TCP_TO_FLANGE = np.array(
    [
        [1.0, 0.0, 0.0, -0.093575],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

_NORMALIZER_PATH = Path(__file__).parent.parent / "normalizer_stats/bridge_orig_lerobot_stats.json"


class BridgeOrigLeRobotDataset(ActionBaseDataset):
    """Bridge Orig dataset with 10D cartesian actions:

        [pos_delta(3), rot6d_delta(6), gripper(1)]

    Uses a single ``image_0`` ego-view video, backward-framewise rot6d actions,
    and quantile normalization.
    """


    def __init__(
        self,
        root: str,
        fps: float = 5.0,
        chunk_length: int = 16,
        mode: str = "joint",
        pose_convention: PoseConvention = "backward_framewise",
        tolerance_s: float = 1e-4,
        viewpoint: Viewpoint = "ego_view",
        action_normalization: str | None = "quantile",
        sample_stride: int = 1,
    ) -> None:
        if viewpoint != "ego_view":
            raise NotImplementedError("This minimal Bridge dataset only supports ego_view.")
        super().__init__(
            root=root,
            domain_name="bridge_orig_lerobot",
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention=pose_convention,
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            sample_stride=sample_stride,
        )

    @property
    def action_dim(self) -> int:
        return 10

    def _action_spec(self) -> ActionSpec:
        return build_action_spec(Pos(), Rot("rot6d"), Gripper())

    @classmethod
    def _stats_path(cls) -> Path:
        return _NORMALIZER_PATH

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx)
        first_row = self._rows[idx]
        episode = self._episodes[int(first_row["episode_index"])]

        row_idx = idx * self._sample_stride
        observation_rows = self._rows[row_idx : row_idx + self._chunk_length + 1]
        action_rows = observation_rows[: self._chunk_length]

        video = self._load_video(episode, observation_rows)
        raw_action, initial_pose = self._build_raw_action(observation_rows, action_rows)
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
            self._video_path(episode, _IMAGE_FEATURE),
            [float(episode.get(f"videos/{_IMAGE_FEATURE}/from_timestamp", 0.0)) + ts for ts in timestamps],
            self._tolerance_s,
        )

    def _build_raw_action(
        self,
        observation_rows: list[dict[str, Any]],
        action_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = np.asarray([row[_STATE_FEATURE] for row in observation_rows], dtype=np.float32)
        poses_abs = build_abs_pose_from_components(state[:, 0:3], state[:, 3:6], "euler_xyz")

        poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ _DEFAULT_ROTATION.astype(poses_abs.dtype)
        poses_abs = poses_abs @ _TCP_TO_FLANGE.astype(poses_abs.dtype)
        poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ _BRIDGE_TO_OPENCV.astype(poses_abs.dtype)

        initial_pose = torch.from_numpy(poses_abs[0].copy()).float()
        poses_rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention=self._pose_convention)
        gripper = np.asarray([row[_ACTION_FEATURE][6] for row in action_rows], dtype=np.float32).reshape(-1, 1)
        action = np.concatenate([poses_rel[-self._chunk_length :], gripper[-self._chunk_length :]], axis=-1)
        return torch.from_numpy(action).float(), initial_pose

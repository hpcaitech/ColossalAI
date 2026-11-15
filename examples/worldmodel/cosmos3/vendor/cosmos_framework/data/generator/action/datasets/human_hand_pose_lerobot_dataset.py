# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Bimanual human hand-pose dataset in LeRobot v3 format."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.datasets.video_utils import decode_video_frames

from cosmos_framework.data.generator.action.utils.action_spec import ActionSpec, Pos, Rot, build_action_spec
from cosmos_framework.data.generator.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.generator.action.utils.pose_utils import build_abs_pose_from_components, pose_abs_to_rel

PoseConvention = Literal["backward_framewise"]
Viewpoint = Literal["ego_view"]

_HAND_RIGHT_POSITION_KEY = "observation.state.hand_right_cam"
_HAND_RIGHT_ROTATION_KEY = "observation.state.hand_right_cam_rotation"
_HAND_LEFT_POSITION_KEY = "observation.state.hand_left_cam"
_HAND_LEFT_ROTATION_KEY = "observation.state.hand_left_cam_rotation"
_CAM_POSITION_KEY = "observation.state.camera_position"
_CAM_ROTATION_KEY = "observation.state.camera_rotation"
_IMAGE_FEATURE = "observation.images.main"

_NUM_JOINTS = 21
_WRIST_JOINT_IDX = 0
_FINGERTIP_JOINT_IDXS = (4, 8, 12, 16, 20)
_RAW_ACTION_DIM = 57
_NORMALIZER_PATH = Path(__file__).parent.parent / "normalizer_stats/human_hand_pose_lerobot_stats.json"

# Rotate the source wrist frames into the unified convention:
# X = thumb-to-pinky, Y = outward palm normal, Z = wrist-to-fingertips.
_WRIST_FRAME_ALIGNMENT = np.array(
    [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)


class HumanHandPoseLeRobotDataset(ActionBaseDataset):
    """Bimanual human hand-pose forward-dynamics data.

    The 57D action layout is
    ``[camera(9), right_wrist(9), right_fingertips(15),
    left_wrist(9), left_fingertips(15)]``. Camera and wrist poses use
    framewise 3D translation plus rot6d deltas. Each hand's five fingertip
    positions are expressed in that frame's aligned wrist coordinate system.

    Source video and pose annotations are sampled at 30 FPS by default and
    decoded at 15 FPS for Cosmos3-Nano, yielding 17 video frames and 16 action
    transitions for the default chunk.
    """

    def __init__(
        self,
        root: str,
        fps: float = 15.0,
        chunk_length: int = 16,
        mode: str = "forward_dynamics",
        pose_convention: PoseConvention = "backward_framewise",
        tolerance_s: float = 2e-4,
        viewpoint: Viewpoint = "ego_view",
        action_normalization: str | None = "quantile",
        sample_stride: int = 1,
        image_key: str = _IMAGE_FEATURE,
    ) -> None:
        if viewpoint != "ego_view":
            raise NotImplementedError("Human hand-pose data only supports ego_view.")
        super().__init__(
            root=root,
            domain_name="hand_pose",
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention=pose_convention,
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            sample_stride=sample_stride,
        )
        source_fps = float(self._info["fps"])
        source_stride = source_fps / self._fps
        if not source_stride.is_integer():
            raise ValueError(f"Source FPS {source_fps} must be an integer multiple of target FPS {self._fps}.")
        self._source_stride = int(source_stride)
        self._image_key = image_key
        required_source_steps = self._source_stride * self._chunk_length
        self._valid_starts: list[int] = []
        episode_start = 0
        while episode_start < len(self._rows):
            episode_index = int(self._rows[episode_start]["episode_index"])
            episode_end = episode_start + 1
            while episode_end < len(self._rows) and int(self._rows[episode_end]["episode_index"]) == episode_index:
                episode_end += 1
            self._valid_starts.extend(
                range(episode_start, max(episode_start, episode_end - required_source_steps), self._sample_stride)
            )
            episode_start = episode_end
        subtasks_path = self._root / "meta" / "subtasks.parquet"
        self._subtasks = (
            {int(row["subtask_index"]): str(row["subtask"]) for row in pq.read_table(subtasks_path).to_pylist()}
            if subtasks_path.exists()
            else {}
        )

    @property
    def action_dim(self) -> int:
        return _RAW_ACTION_DIM

    def _action_spec(self) -> ActionSpec:
        return build_action_spec(
            Pos(prefix="camera"),
            Rot("rot6d", prefix="camera"),
            Pos(prefix="right_wrist"),
            Rot("rot6d", prefix="right_wrist"),
            Pos(dim=15, prefix="right_fingertip"),
            Pos(prefix="left_wrist"),
            Rot("rot6d", prefix="left_wrist"),
            Pos(dim=15, prefix="left_fingertip"),
        )

    @classmethod
    def _stats_path(cls) -> Path:
        return _NORMALIZER_PATH

    def __len__(self) -> int:
        return len(self._valid_starts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        start = self._valid_starts[int(idx)]
        stop = start + self._source_stride * self._chunk_length + 1
        rows = self._rows[start : stop : self._source_stride]
        if len(rows) != self._chunk_length + 1:
            raise IndexError(f"Incomplete hand-pose window at index {idx}.")
        episode_index = int(rows[0]["episode_index"])
        if any(int(row["episode_index"]) != episode_index for row in rows):
            raise IndexError(f"Hand-pose window at index {idx} crosses an episode boundary.")

        episode = self._episodes[episode_index]
        video = self._load_video(episode, rows)
        raw_action = self._build_raw_action(rows)
        subtask_index = int(rows[0].get("subtask_index", -1))
        task = self._tasks[int(rows[0]["task_index"])]
        caption = self._subtasks.get(subtask_index, task)
        ai_caption = random.choice([part.strip() for part in caption.split(" | ") if part.strip()] or [caption])

        result = self._build_result(mode=mode, video=video, action=raw_action, ai_caption=ai_caption)
        if self.action_normalization is not None:
            result["action"] = result["action"].clamp(-1.0, 1.0)
        return result

    def _load_video(self, episode: dict[str, Any], rows: list[dict[str, Any]]) -> torch.Tensor:
        timestamps = [float(row["timestamp"]) for row in rows]
        from_timestamp = float(episode.get(f"videos/{self._image_key}/from_timestamp", 0.0))
        return decode_video_frames(
            self._video_path(episode, self._image_key),
            [from_timestamp + timestamp for timestamp in timestamps],
            self._tolerance_s,
        )

    @staticmethod
    def _finger_positions_in_wrist_frame(position_data: np.ndarray, wrist_poses: np.ndarray) -> np.ndarray:
        future_positions = position_data[1:].reshape(-1, _NUM_JOINTS, 3)
        fingertips = future_positions[:, _FINGERTIP_JOINT_IDXS, :]
        fingertips_h = np.concatenate(
            [fingertips, np.ones((*fingertips.shape[:-1], 1), dtype=np.float32)],
            axis=-1,
        )
        wrist_inv = np.linalg.inv(wrist_poses[1:])
        fingertips_wrist = np.einsum("tij,tnj->tni", wrist_inv, fingertips_h)[..., :3]
        return fingertips_wrist.reshape(len(future_positions), -1)

    def _build_raw_action(self, rows: list[dict[str, Any]]) -> torch.Tensor:
        def values(key: str) -> np.ndarray:
            return np.asarray([row[key] for row in rows], dtype=np.float32)

        camera_pose = build_abs_pose_from_components(values(_CAM_POSITION_KEY), values(_CAM_ROTATION_KEY), "quat_xyzw")

        right_positions = values(_HAND_RIGHT_POSITION_KEY)
        right_rotations = values(_HAND_RIGHT_ROTATION_KEY).reshape(-1, _NUM_JOINTS, 4)
        left_positions = values(_HAND_LEFT_POSITION_KEY)
        left_rotations = values(_HAND_LEFT_ROTATION_KEY).reshape(-1, _NUM_JOINTS, 4)

        right_wrist_camera = (
            build_abs_pose_from_components(right_positions[:, :3], right_rotations[:, _WRIST_JOINT_IDX], "quat_xyzw")
            @ _WRIST_FRAME_ALIGNMENT
        )
        left_wrist_camera = (
            build_abs_pose_from_components(left_positions[:, :3], left_rotations[:, _WRIST_JOINT_IDX], "quat_xyzw")
            @ _WRIST_FRAME_ALIGNMENT
        )

        right_wrist_world = camera_pose @ right_wrist_camera
        left_wrist_world = camera_pose @ left_wrist_camera
        action = np.concatenate(
            [
                pose_abs_to_rel(camera_pose, rotation_format="rot6d", pose_convention=self._pose_convention),
                pose_abs_to_rel(right_wrist_world, rotation_format="rot6d", pose_convention=self._pose_convention),
                self._finger_positions_in_wrist_frame(right_positions, right_wrist_camera),
                pose_abs_to_rel(left_wrist_world, rotation_format="rot6d", pose_convention=self._pose_convention),
                self._finger_positions_in_wrist_frame(left_positions, left_wrist_camera),
            ],
            axis=-1,
        )
        if action.shape != (self._chunk_length, _RAW_ACTION_DIM):
            raise ValueError(
                f"Expected hand-pose action shape {(self._chunk_length, _RAW_ACTION_DIM)}, got {action.shape}."
            )
        return torch.from_numpy(action).float()


__all__ = ["HumanHandPoseLeRobotDataset"]

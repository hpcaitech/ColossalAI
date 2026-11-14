# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""AgiBotWorld-Beta LeRobot dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.agibot_fk import (
    AGIBOT_WORLD_GRIPPER_TO_OPENCV_BY_WRIST,
    apply_agibot_gripper_to_opencv,
    apply_robot_base_motion_to_poses,
    compute_fk_transforms_batch,
    convert_gripper_state_to_open_fraction,
)
from cosmos_framework.data.generator.action.utils.action_spec import ActionSpec, Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.generator.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.generator.action.utils.pose_utils import pose_abs_to_rel

PoseConvention = Literal["backward_framewise"]
Viewpoint = Literal["concat_view", "ego_view"]

_HEAD_KEY = "observation.images.head"
_HAND_LEFT_KEY = "observation.images.hand_left"
_HAND_RIGHT_KEY = "observation.images.hand_right"
_CONCAT_KEY = "observation.images.video_concat_view"

_EFFECTOR_KEY = "observation.states.effector.position"
_JOINT_KEY = "observation.states.joint.position"
_HEAD_STATE_KEY = "observation.states.head.position"
_WAIST_KEY = "observation.states.waist.position"
_ROBOT_POSITION_KEY = "observation.states.robot.position"
_ROBOT_ORIENTATION_KEY = "observation.states.robot.orientation"

_NORMALIZER_PATH = Path(__file__).parent.parent / "normalizer_stats/agibotworld_beta_lerobot_stats.json"


def _split_task_for_caption(task: str) -> tuple[str, str]:
    ai_caption, separator, debug_caption = task.partition("|")
    if not separator:
        return task.strip(), ""
    return ai_caption.strip(), debug_caption.strip()


def _assemble_agibot_world_state(
    effector_pos: np.ndarray,
    joint_pos: np.ndarray,
    head_pos: np.ndarray,
    waist_pos: np.ndarray,
) -> np.ndarray:
    """Assemble standard 20D gripper state from Beta decomposed fields."""

    body_head = np.stack(
        [head_pos[:, 0], head_pos[:, 1], waist_pos[:, 0], waist_pos[:, 1]],
        axis=-1,
    )
    return np.concatenate([joint_pos, effector_pos, body_head], axis=-1).astype(np.float32, copy=False)


def _compute_idle_frames_agibot(action: torch.Tensor) -> int:
    """Small local idle-frame helper for the 29D AgiBot FK layout.

    The shared `compute_idle_frames` expects one rotation group after each
    position block; AgiBot's action spec has three such groups plus grippers.
    For cookbook inference, idle frames are metadata only, so this conservative
    implementation marks the initial low-motion streak length.
    """

    if action.numel() == 0:
        return 0
    abs_action = action.detach().abs()
    motion = torch.cat(
        [
            abs_action[:, 0:3],
            abs_action[:, 9:12],
            abs_action[:, 18:21],
            abs_action[:, 18:19].diff(dim=0, prepend=abs_action[0:1, 18:19]),
            abs_action[:, 28:29].diff(dim=0, prepend=abs_action[0:1, 28:29]),
        ],
        dim=-1,
    ).amax(dim=-1)
    below = motion < 1e-3
    count = 0
    for value in below.tolist():
        if not value:
            break
        count += 1
    return count


class AgiBotWorldBetaLeRobotDataset(ActionBaseDataset):
    """AgiBotWorld-Beta dataset with FK-pose 29D actions.

    Action layout matches the AgiBot World gripper normalizer:

        [head_pos+rot6d(9), right_pos+rot6d(9), right_gripper(1),
         left_pos+rot6d(9), left_gripper(1)]

    The local cookbook asset provides head, left wrist, and right wrist videos.
    By default this wrapper uses `concat_view`: head view on top, left/right
    wrist views resized and concatenated on the bottom.
    """


    def __init__(
        self,
        root: str,
        fps: float = 10.0,
        chunk_length: int = 16,
        mode: str = "joint",
        pose_convention: PoseConvention = "backward_framewise",
        tolerance_s: float = 3e-4,
        viewpoint: Viewpoint = "concat_view",
        action_normalization: str | None = "quantile",
        sample_stride: int = 1,
    ) -> None:
        if viewpoint not in ("concat_view", "ego_view"):
            raise NotImplementedError("Supported viewpoints are concat_view and ego_view.")
        super().__init__(
            root=root,
            domain_name="agibotworld",
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention=pose_convention,
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            sample_stride=sample_stride,
        )
        self._rows_by_episode: dict[int, list[dict[str, Any]]] = {}
        for row in self._rows:
            self._rows_by_episode.setdefault(int(row["episode_index"]), []).append(row)
        self._timestamps_by_episode = {
            episode_id: np.asarray([float(row["timestamp"]) for row in rows], dtype=np.float64)
            for episode_id, rows in self._rows_by_episode.items()
        }

    @property
    def action_dim(self) -> int:
        return 29

    def _action_spec(self) -> ActionSpec:
        return build_action_spec(
            Pos(prefix="head"),
            Rot("rot6d", prefix="head"),
            Pos(prefix="right"),
            Rot("rot6d", prefix="right"),
            Gripper(prefix="right"),
            Pos(prefix="left"),
            Rot("rot6d", prefix="left"),
            Gripper(prefix="left"),
        )

    @classmethod
    def _stats_path(cls) -> Path:
        return _NORMALIZER_PATH

    def _compute_idle_frames(self, action: torch.Tensor) -> int:
        return _compute_idle_frames_agibot(action)

    def __len__(self) -> int:
        return max(0, (len(self._rows) - self._chunk_length + self._sample_stride - 1) // self._sample_stride)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        row_idx = int(idx) * self._sample_stride
        start_row = self._rows[row_idx]
        observation_rows = self._select_observation_rows(start_row)
        episode = self._episodes[int(observation_rows[0]["episode_index"])]
        task = self._tasks[int(observation_rows[0]["task_index"])]
        ai_caption, debug_caption = _split_task_for_caption(task)

        video = self._load_video(episode, observation_rows)
        action, extras = self._build_fk_action(observation_rows)
        if self._viewpoint == "concat_view":
            extras["additional_view_description"] = (
                "The top row shows the head-mounted camera view looking down at the workspace. "
                "The bottom row contains two horizontally concatenated wrist-mounted camera views: "
                "the left hand camera on the left and the right hand camera on the right."
            )
        if debug_caption:
            extras["debug_caption"] = debug_caption

        return self._build_result(
            mode=mode,
            video=video,
            action=action,
            ai_caption=ai_caption,
            action_spec_names=self.action_names,
            **extras,
        )

    def _select_observation_rows(self, start_row: dict[str, Any]) -> list[dict[str, Any]]:
        """Select T+1 rows at this wrapper's target FPS within one episode."""

        episode_id = int(start_row["episode_index"])
        rows = self._rows_by_episode[episode_id]
        timestamps = self._timestamps_by_episode[episode_id]
        start_frame = int(start_row["frame_index"])
        start_ts = float(start_row["timestamp"])
        target_ts = start_ts + np.arange(self._chunk_length + 1, dtype=np.float64) / self._fps
        indices = np.searchsorted(timestamps, target_ts, side="left")
        indices = np.minimum(indices, len(rows) - 1)
        prev = np.maximum(indices - 1, 0)
        choose_prev = np.abs(timestamps[prev] - target_ts) <= np.abs(timestamps[indices] - target_ts)
        indices = np.where(choose_prev, prev, indices)
        if int(indices[-1]) <= start_frame:
            raise IndexError(f"Could not select {self._chunk_length + 1} frames from episode {episode_id} at fps={self._fps}.")
        return [rows[int(i)] for i in indices]

    def _load_video(self, episode: dict[str, Any], observation_rows: list[dict[str, Any]]) -> torch.Tensor:
        if self._viewpoint == "ego_view":
            return self._load_video_key(episode, observation_rows, _HEAD_KEY)

        # Prefer a pre-rendered concat view if present. The local asset includes
        # metadata for this key but not the public mp4, so the fallback composes
        # it from the three camera streams.
        concat_path = self._video_path(episode, _CONCAT_KEY)
        if concat_path.exists():
            return self._load_video_key(episode, observation_rows, _CONCAT_KEY)
        top = self._load_video_key(episode, observation_rows, _HEAD_KEY)
        left = self._load_video_key(episode, observation_rows, _HAND_LEFT_KEY)
        right = self._load_video_key(episode, observation_rows, _HAND_RIGHT_KEY)
        return self._compose_multi_view(top, left, right)

    def _load_video_key(self, episode: dict[str, Any], observation_rows: list[dict[str, Any]], key: str) -> torch.Tensor:
        # lerobot is a heavy, optional ("train" extra) dependency; import lazily.
        from lerobot.datasets.video_utils import decode_video_frames

        timestamps = [float(row["timestamp"]) for row in observation_rows]
        return decode_video_frames(
            self._video_path(episode, key),
            [float(episode.get(f"videos/{key}/from_timestamp", 0.0)) + ts for ts in timestamps],
            self._tolerance_s,
        )

    def _compose_multi_view(self, top: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        # Inputs are [T,C,H,W] float tensors in [0,1].
        _, _, h_top, w_top = top.shape
        half_h, half_w = h_top // 2, w_top // 2
        left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)
        bottom = torch.cat([left, right], dim=-1)
        return torch.cat([top, bottom], dim=-2)

    def _build_fk_action(self, rows: list[dict[str, Any]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        effector_pos = np.asarray([row[_EFFECTOR_KEY] for row in rows], dtype=np.float32)
        joint_pos = np.asarray([row[_JOINT_KEY] for row in rows], dtype=np.float32)
        head_pos = np.asarray([row[_HEAD_STATE_KEY] for row in rows], dtype=np.float32)
        waist_pos = np.asarray([row[_WAIST_KEY] for row in rows], dtype=np.float32)
        robot_pos = np.asarray([row[_ROBOT_POSITION_KEY] for row in rows], dtype=np.float32)
        robot_quat = np.asarray([row[_ROBOT_ORIENTATION_KEY] for row in rows], dtype=np.float32)
        states_np = _assemble_agibot_world_state(effector_pos, joint_pos, head_pos, waist_pos)

        native_fk = compute_fk_transforms_batch(states_np, "agibot_world_gripper")
        native_fk = apply_robot_base_motion_to_poses(native_fk, robot_pos, robot_quat)
        fk = apply_agibot_gripper_to_opencv(native_fk, AGIBOT_WORLD_GRIPPER_TO_OPENCV_BY_WRIST)

        head_rel = pose_abs_to_rel(fk["head_camera"], rotation_format="rot6d", pose_convention=self._pose_convention)
        right_rel = pose_abs_to_rel(fk["right_wrist"], rotation_format="rot6d", pose_convention=self._pose_convention)
        left_rel = pose_abs_to_rel(fk["left_wrist"], rotation_format="rot6d", pose_convention=self._pose_convention)
        right_gripper = convert_gripper_state_to_open_fraction(effector_pos[1:, 1:2])
        left_gripper = convert_gripper_state_to_open_fraction(effector_pos[1:, 0:1])
        action_np = np.concatenate([head_rel, right_rel, right_gripper, left_rel, left_gripper], axis=-1).astype(
            np.float32
        )
        extras = {
            "initial_pose": torch.from_numpy(fk["head_camera"][0].copy()).float(),
            "initial_pose_right": torch.from_numpy(fk["right_wrist"][0].copy()).float(),
            "initial_pose_left": torch.from_numpy(fk["left_wrist"][0].copy()).float(),
        }
        return torch.from_numpy(action_np).float(), extras

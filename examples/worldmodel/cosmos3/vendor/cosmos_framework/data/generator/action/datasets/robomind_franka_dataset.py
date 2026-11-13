# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboMIND Franka LeRobot dataset — single-arm and dual-arm variants."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.action_normalization import load_action_stats
from cosmos_framework.data.generator.action.utils.action_spec import ActionSpec, Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.generator.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.generator.action.utils.pose_utils import (
    build_abs_pose_from_components,
    pose_abs_to_rel,
)

PoseConvention = Literal["backward_framewise"]
Viewpoint = Literal["concat_view", "third_person_view"]

# Dual-arm uses camera_front as the top view; single-arm uses camera_top.
_IMAGE_FEATURES_DUAL = {
    "top": "observation.images.camera_front",
    "left": "observation.images.camera_left",
    "right": "observation.images.camera_right",
}
_IMAGE_FEATURES_SINGLE = {
    "top": "observation.images.camera_top",
    "left": "observation.images.camera_left",
    "right": "observation.images.camera_right",
}
_STATE_FEATURE = "observation.states.end_effector"
_ACTION_FEATURE = "actions.joint_position"

# 90-degree clockwise rotation about the Z axis in the local frame. This matches
# the production RoboMIND Franka wrapper conversion to OpenCV coordinates.
_ROBOMIND_FRANKA_TO_OPENCV: np.ndarray = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

_NORMALIZER_PATH_DUAL = Path(__file__).parent.parent / "normalizer_stats/robomind_franka_dual_stats.json"
_NORMALIZER_PATH_SINGLE = Path(__file__).parent.parent / "normalizer_stats/robomind_franka_stats.json"


def _dual_arm_action_spec():
    return build_action_spec(
        Pos(prefix="left"),
        Rot("rot6d", prefix="left"),
        Gripper(prefix="left"),
        Pos(prefix="right"),
        Rot("rot6d", prefix="right"),
        Gripper(prefix="right"),
    )


class RoboMINDFrankaDataset(ActionBaseDataset):
    """RoboMIND Franka dataset — single-arm (10D) or dual-arm (20D) cartesian actions.

    Single-arm (``robomind-franka``): ``[pos_delta(3), rot6d_delta(6), gripper(1)]``
    Dual-arm (``robomind-franka-dual``):
        ``[left_pos(3), left_rot6d(6), left_gripper(1),
           right_pos(3), right_rot6d(6), right_gripper(1)]``
    """

    _SUPPORTED_EMBODIMENTS = ("robomind-franka", "robomind-franka-dual")

    def __init__(
        self,
        root: str,
        fps: float = 10.0,
        chunk_length: int = 16,
        mode: str = "joint",
        embodiment_type: str = "robomind-franka-dual",
        pose_convention: PoseConvention = "backward_framewise",
        tolerance_s: float = 1e-4,
        viewpoint: Viewpoint = "concat_view",
        action_normalization: str | None = "quantile",
        sample_stride: int = 1,
    ) -> None:
        if embodiment_type not in self._SUPPORTED_EMBODIMENTS:
            raise ValueError(
                f"RoboMINDFrankaDataset only supports {self._SUPPORTED_EMBODIMENTS}, "
                f"got embodiment_type={embodiment_type!r}."
            )
        if viewpoint not in ("concat_view", "third_person_view"):
            raise NotImplementedError(f"RoboMINDFrankaDataset does not support viewpoint={viewpoint!r}.")
        self._embodiment_type = embodiment_type
        self._image_features = (
            _IMAGE_FEATURES_SINGLE if embodiment_type == "robomind-franka" else _IMAGE_FEATURES_DUAL
        )
        super().__init__(
            root=root,
            domain_name=embodiment_type,
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
        return 10 if self._embodiment_type == "robomind-franka" else 20

    def _action_spec(self) -> ActionSpec:
        if self._embodiment_type == "robomind-franka":
            return build_action_spec(Pos(), Rot("rot6d"), Gripper())
        return _dual_arm_action_spec()

    @classmethod
    def _stats_path(cls) -> Path:
        # Class-level default (no instance to branch on) — matches the
        # constructor's default embodiment_type ("robomind-franka-dual").
        return _NORMALIZER_PATH_DUAL

    def load_action_stats(self) -> dict[str, torch.Tensor]:
        """Instance-aware override: respects ``self._embodiment_type``.

        The inherited classmethod always resolves via ``_stats_path()``, which
        has no instance to branch on and is hardcoded to the dual-arm file. A
        single-arm instance calling ``load_action_stats()`` would otherwise
        silently get 20D dual-arm stats instead of its own 10D stats.
        """
        path = _NORMALIZER_PATH_DUAL if self._embodiment_type == "robomind-franka-dual" else _NORMALIZER_PATH_SINGLE
        return {
            key: torch.from_numpy(value).float()
            for key, value in load_action_stats(str(path)).items()
        }

    def _load_norm_stats(self) -> dict[str, torch.Tensor]:
        if self._norm_stats is None:
            self._norm_stats = self.load_action_stats()
        return self._norm_stats

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx)
        row_idx = idx * self._sample_stride
        observation_rows = self._rows[row_idx : row_idx + self._chunk_length + 1]
        action_rows = observation_rows[: self._chunk_length]

        episode = self._episodes[int(observation_rows[0]["episode_index"])]
        task = self._tasks[int(observation_rows[0]["task_index"])]
        ai_caption = random.choice([part.strip() for part in task.split(" | ") if part.strip()] or [task])

        if self._viewpoint == "concat_view":
            video = self._load_concat_video(episode, observation_rows)
            if self._embodiment_type == "robomind-franka":
                view_desc = (
                    "The top row shows a third-person perspective looking towards the single-arm Franka robot from above. "
                    "The bottom-left view looks at the scene from the left side, and the bottom-right view looks at the scene from the right side."
                )
            else:
                view_desc = (
                    "The top row shows a third-person perspective looking towards the dual-arm Franka robot from the front. "
                    "The bottom-left view looks at the scene from the left side, and the bottom-right view looks at the scene from the right side."
                )
        else:
            video = self._load_single_video(episode, observation_rows)
            view_desc = None

        if self._embodiment_type == "robomind-franka":
            raw_action, initial_pose = self._build_raw_action_single(observation_rows, action_rows)
            extras: dict[str, Any] = {"initial_pose": initial_pose}
        else:
            raw_action, initial_pose_left, initial_pose_right = self._build_raw_action_dual(observation_rows, action_rows)
            extras = {"initial_pose": initial_pose_left, "initial_pose_right": initial_pose_right}

        if view_desc is not None:
            extras["additional_view_description"] = view_desc

        return self._build_result(
            mode=mode,
            video=video,
            action=raw_action,
            ai_caption=ai_caption,
            **extras,
        )

    def _load_single_video(
        self,
        episode: dict[str, Any],
        observation_rows: list[dict[str, Any]],
    ) -> torch.Tensor:
        # lerobot is a heavy, optional ("train" extra) dependency; import lazily.
        from lerobot.datasets.video_utils import decode_video_frames

        video_key = self._image_features["top"]
        timestamps = [float(row["timestamp"]) for row in observation_rows]
        return decode_video_frames(
            self._video_path(episode, video_key),
            [float(episode.get(f"videos/{video_key}/from_timestamp", 0.0)) + ts for ts in timestamps],
            self._tolerance_s,
        )

    def _load_concat_video(
        self,
        episode: dict[str, Any],
        observation_rows: list[dict[str, Any]],
    ) -> torch.Tensor:
        # lerobot is a heavy, optional ("train" extra) dependency; import lazily.
        from lerobot.datasets.video_utils import decode_video_frames

        timestamps = [float(row["timestamp"]) for row in observation_rows]
        frames_by_view = {
            name: decode_video_frames(
                self._video_path(episode, video_key),
                [float(episode.get(f"videos/{video_key}/from_timestamp", 0.0)) + ts for ts in timestamps],
                self._tolerance_s,
            )
            for name, video_key in self._image_features.items()
        }

        top = frames_by_view["top"]
        left = frames_by_view["left"]
        right = frames_by_view["right"]
        _, _, h_top, w_top = top.shape
        half_h, half_w = h_top // 2, w_top // 2
        left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)
        bottom = torch.cat([left, right], dim=-1)
        return torch.cat([top, bottom], dim=-2)

    def _build_relative_poses(
        self,
        positions: np.ndarray,
        euler_xyz: np.ndarray,
    ) -> tuple[np.ndarray, torch.Tensor]:
        poses_abs = build_abs_pose_from_components(positions, euler_xyz, "euler_xyz")
        poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ _ROBOMIND_FRANKA_TO_OPENCV
        initial_pose = torch.from_numpy(poses_abs[0].copy()).float()
        poses_rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention=self._pose_convention)
        return poses_rel, initial_pose

    def _build_raw_action_single(
        self,
        observation_rows: list[dict[str, Any]],
        action_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = np.asarray([row[_STATE_FEATURE] for row in observation_rows], dtype=np.float32)
        gripper = np.asarray([row[_ACTION_FEATURE] for row in action_rows], dtype=np.float32)
        poses_rel, initial_pose = self._build_relative_poses(state[:, 0:3], state[:, 3:6])
        action = np.concatenate(
            [poses_rel[-self._chunk_length :], 1.0 - gripper[-self._chunk_length :, [7]]],
            axis=-1,
        )  # [T, 10]
        return torch.from_numpy(action).float(), initial_pose

    def _build_raw_action_dual(
        self,
        observation_rows: list[dict[str, Any]],
        action_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = np.asarray([row[_STATE_FEATURE] for row in observation_rows], dtype=np.float32)
        gripper = np.asarray([row[_ACTION_FEATURE] for row in action_rows], dtype=np.float32)

        poses_rel_left, initial_pose_left = self._build_relative_poses(state[:, 0:3], state[:, 3:6])
        poses_rel_right, initial_pose_right = self._build_relative_poses(state[:, 6:9], state[:, 9:12])
        action = np.concatenate(
            [
                poses_rel_left[-self._chunk_length :],
                1.0 - gripper[-self._chunk_length :, [7]],
                poses_rel_right[-self._chunk_length :],
                1.0 - gripper[-self._chunk_length :, [15]],
            ],
            axis=-1,
        )  # [T, 20]
        return torch.from_numpy(action).float(), initial_pose_left, initial_pose_right

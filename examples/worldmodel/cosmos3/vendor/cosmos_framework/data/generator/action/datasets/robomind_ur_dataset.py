# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboMIND UR LeRobot dataset for single-arm UR5e embodiment."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from cosmos_framework.data.generator.action.utils.action_spec import ActionSpec, Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.generator.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.generator.action.utils.pose_utils import pose_abs_to_rel

PoseConvention = Literal["backward_framewise"]
Viewpoint = Literal["third_person_view"]

_IMAGE_FEATURE = "observation.images.camera_top"
_ACTION_JOINT_FEATURE = "actions.joint_position"

# UR EE frame → OpenCV convention rotation (3×3, post-multiplied).
# Identity: attachment_site (quat="-1 1 0 0" in ur5e_robotiq_2f85.xml) already
# satisfies OpenCV convention (z = approach).
_ROBOMIND_UR_TO_OPENCV: np.ndarray = np.eye(3, dtype=np.float32)

_UR5E_ARM_JOINTS = 6  # shoulder_pan … wrist_3
_UR5E_EE_SITE = "attachment_site"  # flange site in ur5e_robotiq_2f85.xml

_MJCF_PATH = Path(__file__).resolve().parent.parent / "robot_assets" / "ur5e_robotiq_2f85.xml"

_NORMALIZER_PATH = Path(__file__).parent.parent / "normalizer_stats/robomind_ur_stats.json"


class RoboMINDURDataset(ActionBaseDataset):
    """RoboMIND UR dataset with 10D cartesian actions:

        [pos_delta(3), rot6d_delta(6), gripper(1)]

    Uses FK on ``actions.joint_position`` (6 arm joints → MuJoCo
    ``attachment_site`` SE(3) pose).  ``observation.states.end_effector`` is
    NOT used — it is recorded incorrectly (constant) in ~89 % of UR episodes;
    ``actions.joint_position`` is valid for 100 % of episodes.

    The sample also includes ``joint_configs``: absolute joint angles ``(T, 7)``
    from ``actions.joint_position[1:T+1]`` for FK-based robot mesh animation.
    """

    def __init__(
        self,
        root: str,
        fps: float = 10.0,
        chunk_length: int = 16,
        mode: str = "joint",
        pose_convention: PoseConvention = "backward_framewise",
        tolerance_s: float = 1e-4,
        viewpoint: Viewpoint = "third_person_view",
        action_normalization: str | None = None,
        sample_stride: int = 1,
    ) -> None:
        if viewpoint != "third_person_view":
            raise NotImplementedError("RoboMINDURDataset only supports third_person_view.")
        super().__init__(
            root=root,
            domain_name="robomind-ur",
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention=pose_convention,
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            sample_stride=sample_stride,
        )
        self._mj_model, self._mj_data, self._ee_site_id = self._init_mujoco()

    @staticmethod
    def _init_mujoco():
        """Load UR5e+Robotiq MuJoCo model (kinematics-only) and locate the EE site.

        Strips all geoms and mesh/texture/material assets from the MJCF via
        ``MjSpec`` before compile so the model loads without mesh files on disk.
        FK only needs the kinematic tree, so ``mj_forward`` + site poses still
        produce identical EE poses.
        """
        import mujoco

        spec = mujoco.MjSpec.from_file(str(_MJCF_PATH))

        def _strip_geoms(body):
            for g in list(body.geoms):
                spec.delete(g)
            for child in body.bodies:
                _strip_geoms(child)

        _strip_geoms(spec.worldbody)
        for m in list(spec.meshes):
            spec.delete(m)
        for t in list(spec.textures):
            spec.delete(t)
        for mat in list(spec.materials):
            spec.delete(mat)

        mj_model = spec.compile()
        mj_data = mujoco.MjData(mj_model)
        ee_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, _UR5E_EE_SITE)
        if ee_site_id < 0:
            raise RuntimeError(f"EE site '{_UR5E_EE_SITE}' not found in {_MJCF_PATH}")
        return mj_model, mj_data, ee_site_id

    def _fk_ee_poses(self, arm_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run MuJoCo FK for T+1 arm configs → EE site positions and rotations.

        Args:
            arm_q: ``(T+1, 6)`` arm joint angles in radians.

        Returns:
            ``(positions (T+1, 3), rotations (T+1, 3, 3))`` in MuJoCo world frame.
        """
        import mujoco

        T1 = len(arm_q)
        positions = np.empty((T1, 3), dtype=np.float32)
        rotations = np.empty((T1, 3, 3), dtype=np.float32)
        for t in range(T1):
            self._mj_data.qpos[:_UR5E_ARM_JOINTS] = arm_q[t]
            mujoco.mj_forward(self._mj_model, self._mj_data)
            positions[t] = self._mj_data.site_xpos[self._ee_site_id]
            rotations[t] = self._mj_data.site_xmat[self._ee_site_id].reshape(3, 3)
        return positions, rotations

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
        row_idx = idx * self._sample_stride
        # T+1 rows: current frame + T future frames (needed for FK EE poses and joint_configs)
        observation_rows = self._rows[row_idx : row_idx + self._chunk_length + 1]

        episode = self._episodes[int(observation_rows[0]["episode_index"])]
        task = self._tasks[int(observation_rows[0]["task_index"])]
        ai_caption = random.choice([part.strip() for part in task.split(" | ") if part.strip()] or [task])

        video = self._load_video(episode, observation_rows)
        action, initial_pose, joint_configs = self._build_raw_action(observation_rows)

        return self._build_result(
            mode=mode,
            video=video,
            action=action,
            ai_caption=ai_caption,
            initial_pose=initial_pose,
            joint_configs=joint_configs,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [T+1, 7]: 6 arm joints + 1 gripper command
        q = np.asarray([row[_ACTION_JOINT_FEATURE] for row in observation_rows], dtype=np.float32)
        T = len(q) - 1

        # FK EE trajectory: T+1 absolute poses from arm joints via MuJoCo
        fk_pos, fk_rot = self._fk_ee_poses(q[:, :_UR5E_ARM_JOINTS])
        poses_abs = np.tile(np.eye(4, dtype=np.float32), (T + 1, 1, 1))
        poses_abs[:, :3, 3] = fk_pos
        poses_abs[:, :3, :3] = fk_rot @ _ROBOMIND_UR_TO_OPENCV

        initial_pose = torch.from_numpy(poses_abs[0].copy()).float()
        poses_rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention=self._pose_convention)

        # Raw UR gripper: 0=open, 1=closed → invert so action convention is 0=closed, 1=open.
        # joint_configs keeps the raw value; FK mesh uses raw * 255 → Robotiq ctrl.
        gripper = torch.from_numpy(1.0 - q[:T, 6:7]).float()
        action = torch.cat([torch.from_numpy(poses_rel).float(), gripper], dim=-1)  # [T, 10]

        # Mesh animation: frames 1..T of joint position (post-action states)
        joint_configs = torch.from_numpy(q[1 : 1 + T].copy())  # [T, 7]

        return action, initial_pose, joint_configs

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""LIBERO LeRobot dataset (frame-wise-relative action policy).

Mirrors ``DROIDLeRobotDataset``: reads the LeRobot parquet directly, windows by
frame index, and decodes video at each frame's REAL timestamp. That makes it
FPS-agnostic — it works with the 10 FPS community ``lerobot/libero_*`` datasets
and a 20 FPS conversion alike, without LeRobot's ``delta_timestamps`` grid (which
rejects any window whose synthetic timestamps don't land on real frames).

Action layout (``frame_wise_relative``): the stored 7D ``action`` is already a
per-frame delta ``[dpos(3), drot_axisangle(3), gripper(1)]``; only the rotation is
re-encoded to the requested ``rotation_space`` -> ``[dpos(3), rot6d(6), gripper(1)]``
(10D for ``6d``).

NOTE on FPS / stats fidelity: the bundled ``quantile_rot`` stats were computed on
a 20 FPS conversion. Per-frame deltas at 10 FPS span 2x the wall-clock motion, so
use a 20 FPS LIBERO dataset (or recompute stats for the dataset's FPS).
Loading/training is correct at any FPS regardless.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.utils.action_spec import ActionSpec, Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.generator.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.generator.action.libero_pose_utils import libero_action_dim, libero_rotation_format
from cosmos_framework.data.generator.action.utils.pose_utils import convert_rotation
from cosmos_framework.utils import log

CameraMode = Literal["image", "wrist_image", "concat_view"]
RotationSpace = Literal["3d", "6d", "9d"]

_ACTION_FEATURE = "action"
_IMAGE_FEATURE = "observation.images.image"
_WRIST_FEATURE = "observation.images.wrist_image"
_STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")
_NORMALIZERS_DIR = Path(__file__).parent.parent / "normalizer_stats"

_VIEWPOINT_BY_CAMERA = {
    "image": "third_person_view",
    "wrist_image": "wrist_view",
    "concat_view": "concat_view",
}


class LIBEROLeRobotDataset(ActionBaseDataset):
    """LIBERO action-policy dataset with frame-wise-relative rot6d actions.

    10D ``[pos_delta(3), rot6d_delta(6), gripper(1)]`` (for ``rotation_space='6d'``),
    ``concat_view`` third-person + wrist video, and ``quantile_rot`` normalization
    against the bundled stats. Reads parquet + decodes video at real timestamps,
    so the requested ``fps`` is metadata only (it sets ``conditioning_fps`` and the
    prompt duration); frame windows always use the data's actual frames.
    """

    def __init__(
        self,
        root: str,
        fps: float = 20.0,
        chunk_length: int = 16,
        mode: str = "wam",
        tolerance_s: float = 1e-4,
        camera_mode: CameraMode = "concat_view",
        image_size: int = 256,
        action_space: str = "frame_wise_relative",
        rotation_space: RotationSpace = "6d",
        pose_coordinate_frame: str = "native",
        embodiment_type: str = "libero",
        action_normalization: str | None = "quantile_rot",
        action_stats_path: str | None = None,
        split: str = "train",
        val_ratio: float = 0.01,
        seed: int = 0,
        sample_stride: int = 1,
    ) -> None:
        if action_space != "frame_wise_relative":
            raise NotImplementedError(
                f"This LIBERO dataset only supports action_space='frame_wise_relative', got {action_space!r}."
            )
        if camera_mode not in _VIEWPOINT_BY_CAMERA:
            raise ValueError(f"Unsupported camera_mode={camera_mode!r}. Use image/wrist_image/concat_view.")
        split = split.lower().strip()
        if split not in {"train", "val", "valid", "validation", "eval", "test", "full"}:
            raise ValueError(f"Unsupported split={split!r}. Use train/val/full.")
        if chunk_length % 4 != 0:
            raise ValueError(f"chunk_length must be divisible by 4, got {chunk_length}.")

        super().__init__(
            root=root,
            domain_name=embodiment_type,
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention="backward_framewise",  # unused for frame_wise deltas; satisfies the base assert
            tolerance_s=tolerance_s,
            viewpoint=_VIEWPOINT_BY_CAMERA[camera_mode],
            # frame_wise_relative ⇔ backward_framewise idle semantics. quantile_rot is a
            # LIBERO convention -> normalize with the "quantile" formula on raw-rotation
            # stats (see _load_norm_stats); pass the method the base will call.
            action_normalization=None if action_normalization is None else "quantile",
            sample_stride=sample_stride,
        )
        # FPS-agnostic loader: trust the dataset's NATIVE fps for conditioning_fps /
        # prompt duration so the metadata is truthful (10 for the public
        # lerobot/libero_*, 20 for a 20 FPS conversion). Frame sampling uses each
        # frame's real timestamp regardless, so the requested ``fps`` is ignored here.
        info_fps = self._info.get("fps")
        if info_fps:
            if int(info_fps) != int(fps):
                log.info(f"Using dataset native fps={info_fps} for conditioning (requested {fps}).")
            self._fps = float(info_fps)
            self._dt = 1.0 / self._fps
        self._camera_mode = camera_mode
        self._image_size = int(image_size)
        self._rotation_space = rotation_space.lower().strip()
        self._pose_coordinate_frame = pose_coordinate_frame
        self._embodiment_type = embodiment_type
        self._requested_normalization = action_normalization
        # quantile_rot normalizes against the raw (un-orthonormalized) rotation stats
        # under "global_raw"; everything else uses "global".
        self._stats_key = "global_raw" if action_normalization == "quantile_rot" else "global"
        self._stats_file = self._resolve_stats_file(action_stats_path)

        if self._camera_mode == "image":
            self._video_keys = [_IMAGE_FEATURE]
        elif self._camera_mode == "wrist_image":
            self._video_keys = [_WRIST_FEATURE]
        else:
            self._video_keys = [_IMAGE_FEATURE, _WRIST_FEATURE]

        # Compact, lazy frame index (mirrors DROIDLeRobotDataset): read only the
        # columns the sample builder needs into contiguous arrays, ordered by global
        # frame index, so DataLoader worker forks share them copy-on-write.
        index_parts, episode_parts, task_parts, ts_parts, action_parts = [], [], [], [], []
        for path in sorted((self._root / "data").glob("chunk-*/file-*.parquet")):
            table = pq.read_table(path, columns=["index", "episode_index", "task_index", "timestamp", _ACTION_FEATURE])
            index_parts.append(table["index"].to_numpy())
            episode_parts.append(table["episode_index"].to_numpy())
            task_parts.append(table["task_index"].to_numpy())
            ts_parts.append(table["timestamp"].to_numpy())
            action_parts.append(np.asarray(table[_ACTION_FEATURE].to_pylist(), dtype=np.float32))
        if not index_parts:
            raise FileNotFoundError(f"No data parquet found under {self._root / 'data'}.")
        order = np.argsort(np.concatenate(index_parts).astype(np.int64), kind="stable")
        self._row_episode = np.concatenate(episode_parts).astype(np.int64)[order]
        self._row_task = np.concatenate(task_parts).astype(np.int64)[order]
        self._row_timestamp = np.concatenate(ts_parts).astype(np.float64)[order]
        self._row_action = np.concatenate(action_parts, axis=0).astype(np.float32)[order]

        assert np.all(np.diff(self._row_episode) >= 0), "episode_index not contiguous after sorting by frame index"
        ep_vals, ep_starts, ep_counts = np.unique(self._row_episode, return_index=True, return_counts=True)

        # Deterministic per-episode train/val split (seeded; same on every rank).
        keep = self._split_episode_ids(ep_vals.tolist(), split, val_ratio, seed)
        kept = np.array([int(v) in keep for v in ep_vals], dtype=bool)
        self._ep_vals = ep_vals.astype(np.int64)[kept]
        self._ep_starts = ep_starts.astype(np.int64)[kept]
        kept_counts = ep_counts.astype(np.int64)[kept]
        # Within-episode windows only: total - n_kept_episodes * chunk_length valid samples.
        self._valid_cum = np.cumsum(np.maximum(0, kept_counts - self._chunk_length)).astype(np.int64)

        log.info(
            f"Loaded LIBERO dataset root={self._root} split={split!r} camera_mode={camera_mode!r} "
            f"fps={self._fps} kept_episodes={len(self._ep_vals)}/{len(ep_vals)} "
            f"valid_indices={int(self._valid_cum[-1]) if self._valid_cum.size else 0}"
        )

    # ---- spec / dims -------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return libero_action_dim(self._rotation_space)

    def _action_spec(self) -> ActionSpec:
        return build_action_spec(Pos(), Rot(libero_rotation_format(self._rotation_space)), Gripper())

    @classmethod
    def _stats_path(cls) -> Path:
        # Base classmethod fallback; the instance uses self._stats_file (which also
        # honors action_stats_path + the rotation/coordinate-frame-specific filename).
        return _NORMALIZERS_DIR / "libero_native_frame_wise_relative_rot6d.json"

    # ---- normalization (nested global/global_raw + quantile_rot) ------------

    def _bundled_stats_filename(self) -> str:
        rotation_suffix = {"3d": "3d", "6d": "rot6d", "9d": "rot9d"}.get(self._rotation_space)
        if rotation_suffix is None:
            raise ValueError(f"Unsupported rotation_space={self._rotation_space!r}.")
        action_space = "frame_wise_relative"
        return f"{self._embodiment_type}_{self._pose_coordinate_frame}_{action_space}_{rotation_suffix}.json"

    def _resolve_stats_file(self, action_stats_path: str | None) -> Path:
        if action_stats_path:
            p = Path(action_stats_path)
            if not p.is_absolute():
                p = _NORMALIZERS_DIR / p.name
            if not p.exists():
                raise FileNotFoundError(f"action_stats_path not found: {action_stats_path!r}")
            return p
        p = _NORMALIZERS_DIR / self._bundled_stats_filename()
        if not p.exists():
            raise FileNotFoundError(
                f"Bundled LIBERO stats not found at {p}. Pass action_stats_path or recompute stats."
            )
        return p

    def _load_norm_stats(self) -> dict[str, torch.Tensor]:
        if self._norm_stats is None:
            raw = json.loads(self._stats_file.read_text())[self._stats_key]
            self._norm_stats = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw.items() if k in _STAT_KEYS}
        return self._norm_stats

    # ---- index helpers -----------------------------------------------------

    @staticmethod
    def _split_episode_ids(ep_ids: list[int], split: str, val_ratio: float, seed: int) -> set[int]:
        if split == "full":
            return set(int(v) for v in ep_ids)
        if not (0.0 < val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}.")
        n_val = max(1, int(round(len(ep_ids) * val_ratio)))
        rng = random.Random(seed)  # identical selection on every rank
        val = set(int(v) for v in rng.sample(list(ep_ids), n_val))
        if split == "train":
            return set(int(v) for v in ep_ids) - val
        return val  # val/valid/validation/eval/test

    def __len__(self) -> int:
        return int(self._valid_cum[-1]) if self._valid_cum.size else 0

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """Per-episode ``(start, length)`` flat-index blocks for
        ``ActionIterableShuffleDataset`` (shuffle block ORDER + shard across
        ranks, sequential within a block)."""
        blocks: list[tuple[int, int]] = []
        prev = 0
        for c in np.asarray(self._valid_cum).tolist():
            c = int(c)
            if c > prev:
                blocks.append((prev, c - prev))
            prev = c
        return blocks

    # ---- sample build ------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Resample a different valid window if a frame fails to decode (bounded retries).
        n = len(self)
        last_err: Exception | None = None
        for _attempt in range(8):
            try:
                return self._build_item(idx)
            except Exception as e:  # noqa: BLE001 — skip past undecodable frames
                last_err = e
                log.warning(f"LIBERO: sample idx={idx} failed to load ({type(e).__name__}: {e}); resampling")
                if n > 0:
                    idx = random.randint(0, n - 1)
        raise RuntimeError(f"LIBERO: failed to load a sample after 8 resamples; last error: {last_err}")

    def _build_item(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx)
        ep = int(np.searchsorted(self._valid_cum, idx, side="right"))
        prev = int(self._valid_cum[ep - 1]) if ep > 0 else 0
        start = int(self._ep_starts[ep]) + (idx - prev)
        episode_index = int(self._ep_vals[ep])
        episode = self._episodes[episode_index]

        stop = start + self._chunk_length + 1
        timestamps = [float(self._row_timestamp[j]) for j in range(start, stop)]
        video = self._load_video(episode, timestamps)

        # frame_wise_relative: chunk per-frame deltas are the stored actions directly.
        raw = self._row_action[start : start + self._chunk_length]  # [chunk, 7]
        action = self._build_frame_wise_action(raw)

        task = self._tasks[int(self._row_task[start])]
        ai_caption = random.choice([p.strip() for p in task.split(" | ") if p.strip()] or [task])

        extras: dict[str, Any] = {}
        if self._camera_mode == "concat_view":
            extras["additional_view_description"] = (
                "The left half shows the third-person view; the right half shows the wrist-mounted camera."
            )
        return self._build_result(mode=mode, video=video, action=action, ai_caption=ai_caption, **extras)

    def _build_frame_wise_action(self, raw: np.ndarray) -> torch.Tensor:
        raw_t = torch.from_numpy(np.ascontiguousarray(raw)).float()  # [chunk, 7]
        translation = raw_t[:, 0:3]
        rotation_matrix = convert_rotation(raw_t[:, 3:6], input_format="axisangle", output_format="matrix")
        rotation = convert_rotation(
            rotation_matrix, input_format="matrix", output_format=libero_rotation_format(self._rotation_space)
        )
        gripper = raw_t[:, 6:7]
        return torch.cat([translation, rotation, gripper], dim=-1)  # [chunk, action_dim]

    def _load_video(self, episode: dict[str, Any], timestamps: list[float]) -> torch.Tensor:
        # lerobot is a heavy, optional ("train" extra) dependency; import lazily.
        from lerobot.datasets.video_utils import decode_video_frames

        frames_by_view = {}
        for key in self._video_keys:
            from_ts = float(episode.get(f"videos/{key}/from_timestamp", 0.0))
            frames = decode_video_frames(
                self._video_path(episode, key),
                [from_ts + ts for ts in timestamps],
                self._tolerance_s,
            )  # [T, C, H, W] in [0, 1]
            frames = self._resize(frames)
            frames_by_view[key] = frames
        if self._camera_mode == "concat_view":
            # third-person (left) + wrist (right), horizontally concatenated -> [T, C, H, 2W]
            return torch.cat([frames_by_view[_IMAGE_FEATURE], frames_by_view[_WRIST_FEATURE]], dim=-1)
        return frames_by_view[self._video_keys[0]]

    def _resize(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[-1] == self._image_size and frames.shape[-2] == self._image_size:
            return frames
        return F.interpolate(frames, size=(self._image_size, self._image_size), mode="bilinear", align_corners=False)

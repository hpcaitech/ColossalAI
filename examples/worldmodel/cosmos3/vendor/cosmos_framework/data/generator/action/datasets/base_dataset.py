# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Abstract base class for Action LeRobot datasets."""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.action_normalization import load_action_stats, normalize_action
from cosmos_framework.data.generator.action.utils.action_spec import ActionSpec
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.utils.pose_utils import compute_idle_frames

_MODE_CHOICES = ("forward_dynamics", "inverse_dynamics", "wam")


class ActionBaseDataset(ABC, Dataset):
    """Abstract base for Action LeRobot datasets.

    Subclasses must implement the abstract methods listed below.
    """

    def __init__(
        self,
        root: str,
        domain_name: str,
        fps: float,
        chunk_length: int,
        mode: str,
        pose_convention: str,
        tolerance_s: float,
        viewpoint: str,
        action_normalization: str | None = "quantile",
        sample_stride: int = 1,
    ) -> None:
        super().__init__()
        if pose_convention != "backward_framewise":
            raise NotImplementedError(f"{type(self).__name__} only supports backward_framewise pose deltas.")

        self._fps = float(fps)
        self._dt = 1.0 / self._fps
        self._chunk_length = int(chunk_length)
        self._sample_stride = int(sample_stride)
        if self._sample_stride < 1:
            raise ValueError(f"sample_stride must be >= 1, got {self._sample_stride}")
        self._mode = mode
        self._pose_convention = pose_convention
        self._tolerance_s = float(tolerance_s)
        self._viewpoint = viewpoint
        self._domain_name = domain_name
        self._domain_id = get_domain_id(domain_name)
        self._action_normalization = action_normalization
        self._norm_stats: dict[str, torch.Tensor] | None = None

        self._root = Path(root)
        self._info = json.loads((self._root / "meta" / "info.json").read_text())
        self._episodes = {
            int(row["episode_index"]): row
            for path in sorted((self._root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
            for row in pq.read_table(path).to_pylist()
        }
        tasks_df = pd.read_parquet(self._root / "meta" / "tasks.parquet")
        # LeRobot v2.x stores task text in a "task" column; v3.0 stores it as the
        # (unnamed) DataFrame index and keeps only "task_index" as a column.
        task_texts = tasks_df["task"] if "task" in tasks_df.columns else tasks_df.index
        self._tasks = {int(task_index): str(task) for task, task_index in zip(task_texts, tasks_df["task_index"])}
        # ``self._rows`` (the flat, index-sorted list of every frame dict) is built
        # lazily on first access — see the ``_rows`` property. Materializing all
        # ~18M frames as Python dicts plus a full sort costs ~13 min and tens of GB;
        # subclasses that build their own compact index (e.g. DROIDLeRobotDataset)
        # never touch it, so they must not pay for it at construction.
        self._rows_cache: list[dict[str, Any]] | None = None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._mode = value

    @property
    def domain_name(self) -> str:
        return self._domain_name

    @property
    def viewpoint(self) -> str:
        return self._viewpoint

    @property
    def domain_id(self) -> int:
        return self._domain_id

    @property
    def action_normalization(self) -> str:
        return self._action_normalization

    @property
    @abstractmethod
    def action_dim(self) -> int: ...

    @abstractmethod
    def _action_spec(self) -> ActionSpec: ...

    @property
    def action_names(self) -> list[str]:
        return self._action_spec().names

    @classmethod
    @abstractmethod
    def _stats_path(cls) -> Path:
        """Return the path to the stats JSON file for this dataset."""
        ...

    @classmethod
    def load_action_stats(cls) -> dict[str, torch.Tensor]:
        """Return action normalization stats for this dataset as torch tensors."""
        return {
            key: torch.from_numpy(value).float() for key, value in load_action_stats(str(cls._stats_path())).items()
        }

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]: ...

    def _compute_idle_frames(self, action: torch.Tensor) -> int:
        return compute_idle_frames(
            action,
            self._action_spec(),
            eps_t=5e-3 / self._fps,
            eps_r=np.deg2rad(1.5) / self._fps,
            eps_g=1e-2,
            joint_threshold=5e-3 / self._fps,
            min_streak=3,
        )

    def _choose_mode(self) -> str:
        if self._mode == "joint":
            return random.choice(_MODE_CHOICES)
        return self._mode

    def _video_path(self, episode: dict[str, Any], video_key: str) -> Path:
        chunk_idx = int(
            episode.get(
                f"videos/{video_key}/chunk_index",
                episode.get(f"videos/{video_key}/episode_chunk", episode.get("data/chunk_index", 0)),
            )
        )
        file_idx = int(
            episode.get(
                f"videos/{video_key}/file_index",
                episode.get(f"videos/{video_key}/episode_file", episode.get("data/file_index", 0)),
            )
        )
        rel = self._info["video_path"].format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
            episode_chunk=chunk_idx,
            episode_file=file_idx,
        )
        return self._root / rel

    def _load_norm_stats(self) -> dict[str, torch.Tensor]:
        if self._norm_stats is None:
            self._norm_stats = self.load_action_stats()
        return self._norm_stats

    def _build_result(
        self,
        *,
        mode: str,
        video: torch.Tensor,
        action: torch.Tensor,
        ai_caption: str,
        **extras: Any,
    ) -> dict[str, Any]:
        idle_frames = self._compute_idle_frames(action)
        # action_normalization=None -> use raw actions (no normalization), e.g. joint_pos.
        if self.action_normalization is None:
            normalized_action = action
        else:
            normalized_action = normalize_action(action, self.action_normalization, self._load_norm_stats())
        formatted_video = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3)
        return {
            "ai_caption": ai_caption,
            "video": formatted_video,
            "action": normalized_action,
            "conditioning_fps": torch.tensor(self._fps, dtype=torch.long),
            "mode": mode,
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": self._viewpoint,
            "idle_frames": torch.tensor(idle_frames, dtype=torch.long),
            **extras,
        }

    @property
    def _rows(self) -> list[dict[str, Any]]:
        """Flat, index-sorted list of every frame dict, built lazily on first access.

        Only datasets that don't build their own compact index (bridge / agibot /
        robomind) touch this; for them it materializes once and caches. Datasets with
        a bespoke index (e.g. DROIDLeRobotDataset) never read it, so they skip the
        ~13 min / tens-of-GB construction entirely.
        """
        if self._rows_cache is None:
            self._rows_cache = sorted(
                (
                    row
                    for path in sorted((self._root / "data").glob("chunk-*/file-*.parquet"))
                    for row in pq.read_table(path).to_pylist()
                ),
                key=lambda row: int(row["index"]),
            )
        return self._rows_cache

    def __len__(self) -> int:
        return max(0, (len(self._rows) - self._chunk_length + self._sample_stride - 1) // self._sample_stride)

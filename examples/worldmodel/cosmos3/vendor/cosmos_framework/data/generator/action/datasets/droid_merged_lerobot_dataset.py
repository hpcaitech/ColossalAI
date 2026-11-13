# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""DROID-Merged LeRobot dataset used by the action FD DROID recipe.

This wrapper keeps all sample parsing, action construction, video loading, and
normalization in ``DROIDLeRobotDataset``. It only registers a Cosmos3-DROID
parent directory (``success/`` + ``failure/``) as a layout the parent class can
consume.
"""

from __future__ import annotations

import os
from pathlib import Path

from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset
from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset_config import (
    ACTION_FEATURES,
    HAS_MULTI_LANGUAGE_ANNOTATIONS,
    IMAGE_FEATURES,
    IS_FLAT_ACTION,
    IS_GRIPPER_ACTION_FLIPPED,
    LEROBOT_ROOTS,
    STATE_FEATURES,
)

_TEMPLATE_VERSION = "droid_plus_lerobot_640x360_20260412"
_PARENT_EE_POSE_ACTION_SPACE = "midtrain"


class DROIDMergedLeRobotDataset(DROIDLeRobotDataset):
    """DROID dataset wrapper for parent roots containing success/failure splits.

    ``DROIDLeRobotDataset`` already supports merged DROID layouts when the root
    basename is known in ``droid_lerobot_dataset_config.py``. This class adds the
    same metadata for arbitrary local parent directories such as
    ``datasets/Cosmos3-DROID`` while delegating all data semantics to the parent.
    """

    def __init__(self, *args, action_space: str = "ee_pose", **kwargs) -> None:
        root = kwargs.get("root")
        if root is None and args:
            root = args[0]
        if root is None:
            raise TypeError("DROIDMergedLeRobotDataset requires a root argument.")

        root_path = Path(root)
        self._register_root_layout(root_path)

        if action_space == "ee_pose":
            # Parent DROIDLeRobotDataset currently names this same 10-D
            # [pos_delta, rot6d_delta, gripper] branch internally.
            action_space = _PARENT_EE_POSE_ACTION_SPACE

        super().__init__(*args, action_space=action_space, **kwargs)

    @classmethod
    def _register_root_layout(cls, root: Path) -> None:
        version = os.path.basename(root)
        if version in LEROBOT_ROOTS:
            return

        roots = cls._discover_split_roots(root)
        LEROBOT_ROOTS[version] = roots
        IMAGE_FEATURES[version] = dict(IMAGE_FEATURES[_TEMPLATE_VERSION])
        STATE_FEATURES[version] = STATE_FEATURES[_TEMPLATE_VERSION]
        ACTION_FEATURES[version] = ACTION_FEATURES[_TEMPLATE_VERSION]
        IS_FLAT_ACTION[version] = IS_FLAT_ACTION[_TEMPLATE_VERSION]
        HAS_MULTI_LANGUAGE_ANNOTATIONS[version] = HAS_MULTI_LANGUAGE_ANNOTATIONS[_TEMPLATE_VERSION]
        IS_GRIPPER_ACTION_FLIPPED[version] = IS_GRIPPER_ACTION_FLIPPED[_TEMPLATE_VERSION]

    @staticmethod
    def _is_lerobot_root(path: Path) -> bool:
        return (path / "meta" / "info.json").is_file()

    @classmethod
    def _discover_split_roots(cls, root: Path) -> list[str] | None:
        if cls._is_lerobot_root(root):
            return None

        discovered: list[str] = []
        for split_name in ("success", "failure"):
            split_root = root / split_name
            if cls._is_lerobot_root(split_root):
                discovered.append(split_name)
                continue
            if split_root.is_dir():
                discovered.extend(
                    f"{split_name}/{path.name}" for path in sorted(split_root.iterdir()) if cls._is_lerobot_root(path)
                )

        if discovered:
            return discovered
        raise FileNotFoundError(
            f"{root} is not a LeRobot root and no success/failure LeRobot splits were found under it."
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

_INSTITUTIONS = [
    "AUTOLab",
    "CLVR",
    "GuptaLab",
    "ILIAD",
    "IPRL",
    "IRIS",
    "PennPAL",
    "RAD",
    "RAIL",
    "REAL",
    "RPL",
    "TRI",
    "WEIRD",
]

LEROBOT_ROOTS = {
    "droid_lerobot_20260115_no_noops": None,
    "droid_plus_lerobot_320x180_20260406_sharded": [f"success/{x}" for x in _INSTITUTIONS]
    + [f"failure/{x}" for x in _INSTITUTIONS],
    "droid_plus_lerobot_320x180_20260406": ["success", "failure"],
    "droid_plus_lerobot_640x360_20260412_sharded": [f"success/{x}" for x in _INSTITUTIONS]
    + [f"failure/{x}" for x in _INSTITUTIONS],
    "droid_plus_lerobot_640x360_20260412": ["success", "failure"],
}

IMAGE_FEATURES = {
    "droid_lerobot_20260115_no_noops": {
        "wrist": "observation.images.wrist_image_left",
        "left": "observation.images.exterior_image_1_left",
        "right": "observation.images.exterior_image_2_left",
    },
    "droid_plus_lerobot_320x180_20260406_sharded": {
        "wrist": "observation.image.wrist_image_left",
        "left": "observation.image.exterior_image_1_left",
        "right": "observation.image.exterior_image_2_left",
    },
    "droid_plus_lerobot_320x180_20260406": {
        "wrist": "observation.image.wrist_image_left",
        "left": "observation.image.exterior_image_1_left",
        "right": "observation.image.exterior_image_2_left",
    },
    "droid_plus_lerobot_640x360_20260412_sharded": {
        "wrist": "observation.image.wrist_image_left",
        "left": "observation.image.exterior_image_1_left",
        "right": "observation.image.exterior_image_2_left",
    },
    "droid_plus_lerobot_640x360_20260412": {
        "wrist": "observation.image.wrist_image_left",
        "left": "observation.image.exterior_image_1_left",
        "right": "observation.image.exterior_image_2_left",
    },
}

STATE_FEATURES = {
    "droid_lerobot_20260115_no_noops": "observation.state",
    "droid_plus_lerobot_320x180_20260406_sharded": "observation.state.cartesian_position",
    "droid_plus_lerobot_320x180_20260406": "observation.state.cartesian_position",
    "droid_plus_lerobot_640x360_20260412_sharded": "observation.state.cartesian_position",
    "droid_plus_lerobot_640x360_20260412": "observation.state.cartesian_position",
}

ACTION_FEATURES = {
    "droid_lerobot_20260115_no_noops": "action",
    "droid_plus_lerobot_320x180_20260406_sharded": "action.gripper_position",
    "droid_plus_lerobot_320x180_20260406": "action.gripper_position",
    "droid_plus_lerobot_640x360_20260412_sharded": "action.gripper_position",
    "droid_plus_lerobot_640x360_20260412": "action.gripper_position",
}

IS_FLAT_ACTION = {
    "droid_lerobot_20260115_no_noops": True,
    "droid_plus_lerobot_320x180_20260406_sharded": False,
    "droid_plus_lerobot_320x180_20260406": False,
    "droid_plus_lerobot_640x360_20260412_sharded": False,
    "droid_plus_lerobot_640x360_20260412": False,
}

HAS_MULTI_LANGUAGE_ANNOTATIONS = {
    "droid_lerobot_20260115_no_noops": False,
    "droid_plus_lerobot_320x180_20260406_sharded": True,
    "droid_plus_lerobot_320x180_20260406": True,
    "droid_plus_lerobot_640x360_20260412_sharded": True,
    "droid_plus_lerobot_640x360_20260412": True,
}

IS_GRIPPER_ACTION_FLIPPED = {
    "droid_lerobot_20260115_no_noops": False,
    "droid_plus_lerobot_320x180_20260406_sharded": True,
    "droid_plus_lerobot_320x180_20260406": True,
    "droid_plus_lerobot_640x360_20260412_sharded": True,
    "droid_plus_lerobot_640x360_20260412": True,
}

_JOINT_ACTION_FEATURE = "action.joint_position"
_JOINT_STATE_FEATURE = "observation.state.joint_positions"
_GRIPPER_STATE_FEATURE = "observation.state.gripper_position"

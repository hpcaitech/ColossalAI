# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Domain ID helpers for cross-embodiment action datasets."""

EMBODIMENT_TO_DOMAIN_ID: dict[str, int] = {
    "no_action": 0,
    "av": 1,
    "camera_pose": 2,
    "hand_pose": 3,
    # WebHumanAction omits the camera prefix from its 48D native hand layout,
    # so it must not reuse hand_pose/embodiment_a's camera-inclusive domain 3 projector.
    "webhumanaction_hand": 31,
    "pusht": 4,
    "libero": 5,
    "umi": 6,
    "bridge_orig_lerobot": 7,
    "droid_lerobot": 8,
    "robomind-franka": 8,  # Both Droid and RoboMIND-Franka are using robotiq and franka
    "embodiment_b": 9,
    "robomind-franka-dual": 12,
    "robomind-ur": 13,
    "agibotworld": 15,
    "embodiment_c_gripper": 15,
    "embodiment_c_gripper_ext": 15,
    "xdof_yam": 16,
    "molmoact2_yam": 16,  # MolmoAct2 uses the same YAM 20D FK action contract
    "abc_yam": 16,  # ABC uses the same YAM 20D FK action contract
    "robotwin": 17,  # RoboTwin dual-arm ALOHA (14D absolute joint_pos)
    # Shares RoboTwin's domain: the same arm, differing only by millimetre link
    # lengths and a 0.9 deg yaw calibration baked into RoboTwin's URDF.
    "robodojo": 17,
    "fractal": 20,
    "drawanything": 21,
    "behavior1k_lerobot": 22,  # BEHAVIOR-1K R1Pro mobile bimanual (23D joint action)
    "maniparena": 23,  # ManipArena x2robot/ex001_6r dual-arm; own 20D EE-direct action projection
    # New dedicated slot (not reusing agibot's domain 15): WebHumanAction body
    # (ego/head+wrists+fingertips, no camera action, 57D) trains its own action2llm/llm2action
    # DomainAwareLinear weights from scratch instead of continuing agibot's.
    "webhumanaction_body": 24,
    "so101-molmo-midtrain-15hz": 25,
    "so100-molmo-midtrain-15hz": 26,
    "so101-bimanual-midtrain-conditional": 27,
    # GenieSim G2A is the simulated counterpart of Embodiment C and emits the
    # same 29-D [head, right arm/gripper, left arm/gripper] contract.
    "geniesim3_g2a": 15,
    "geniesim3_g2a_joint": 28,
    # ManipArena mobile-manipulation children only: 29D head+dual-arm SE(3). A
    # dedicated slot rather than sharing domain 23, because the width and semantics
    # differ from the 20D contract and it trains its own action2llm/llm2action
    # DomainAwareLinear weights. New embodiments append above the maximum; the gaps
    # at 10/11/14/18/19 index retired embodiments and are not reused.
    "maniparena_mobile": 29,
    # RoboCasa PandaOmron mobile manipulation (10/15/20D raw action per
    # ``use_base_action`` / ``base_encoding``); appended above the maximum.
    "robocasa": 30,
    # embodiment_b nvidia-20260828 ingestion: a new one-shot dataset, distinct from
    # "embodiment_b" (domain 9, an earlier unrelated sample drop with its own 30D
    # contract).
    "embodiment_b_20260828": 32,
}


EMBODIMENT_TO_RAW_ACTION_DIM: dict[str, int] = {
    "av": 9,
    "camera_pose": 9,
    "pusht": 2,
    "umi": 10,
    "bridge_orig_lerobot": 10,
    "droid_lerobot": 10,
    "robomind-franka": 10,
    "robomind-franka-dual": 20,
    "robomind-ur": 10,
    "embodiment_b": 30,
    "agibotworld": 29,
    "embodiment_c_gripper": 29,
    "embodiment_c_gripper_ext": 29,
    "webhumanaction_body": 57,  # ego/head(9) + [R_wrist(9)+R_fingertips(15)] + [L_wrist(9)+L_fingertips(15)]
    "xdof_yam": 20,
    "molmoact2_yam": 20,
    "abc_yam": 20,
    "maniparena": 20,  # dual-arm EE: [pos(3)+rot6d(6)+gripper(1)] x 2
    # head(9) + left(9)+grip(1) + right(9)+grip(1); head/left/right, not head/right/left
    "maniparena_mobile": 29,
    "robotwin": 14,  # dual-arm ALOHA: [L 6 joints + 1 gripper, R 6 joints + 1 gripper]
    "robodojo": 14,  # dual-arm ARX-X5: [L 6 joints + 1 gripper, R 6 joints + 1 gripper]
    "fractal": 10,
    "drawanything": 3,
    "behavior1k_lerobot": 23,  # base(3) trunk(4) arms(14) grippers(2)
    "so101-molmo-midtrain-15hz": 10,
    "so100-molmo-midtrain-15hz": 10,
    "so101-bimanual-midtrain-conditional": 20,
    "geniesim3_g2a": 29,
    "geniesim3_g2a_joint": 16,
    "embodiment_b_20260828": 50,
    # NOTE: ``libero`` (7/10/13 depending on ``rotation_space``), ``hand_pose``
    # (variable with ``keypoint_option`` and ``rotation_format``) and ``robocasa``
    # (10 arm-only, 15/20 with the mobile base, per ``use_base_action`` /
    # ``base_encoding``) are absent because their raw width is set per-dataset at
    # construction time. Inference in inverse_dynamics/WAM modes is not supported
    # for these domains until canonical widths are added here.
}


def get_domain_id(embodiment_type: str) -> int:
    """Get the domain ID for a given embodiment type."""
    key = embodiment_type.lower().strip()
    if key not in EMBODIMENT_TO_DOMAIN_ID:
        raise KeyError(
            f"Unknown embodiment type: {embodiment_type!r}. "
            f"Available embodiments: {sorted(EMBODIMENT_TO_DOMAIN_ID.keys())}"
        )
    return EMBODIMENT_TO_DOMAIN_ID[key]


def get_action_dim(embodiment_type: str) -> int:
    """Get the raw action dimension for a given embodiment type."""
    key = embodiment_type.lower().strip()
    if key not in EMBODIMENT_TO_RAW_ACTION_DIM:
        raise KeyError(
            f"Unknown embodiment type: {embodiment_type!r}. "
            f"Available embodiments: {sorted(EMBODIMENT_TO_RAW_ACTION_DIM.keys())}"
        )
    return EMBODIMENT_TO_RAW_ACTION_DIM[key]


def is_valid_domain_name(embodiment_type: str) -> bool:
    """Check if the given embodiment type is recognized."""
    key = embodiment_type.lower().strip()
    return key in EMBODIMENT_TO_RAW_ACTION_DIM

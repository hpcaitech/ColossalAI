# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Central caption-attribute protocols for Action datasets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Domain = Literal["real", "synthetic"]
DomainMappings: dict[Domain, str] = {
    "real": "The video is captured from a real-world environment.",
    "synthetic": "The video is captured from a synthetic environment.",
}

Embodiment = Literal["single_arm", "dual_arm", "human", "camera_only", "vehicle"]
EmbodimentMappings: dict[Embodiment, str] = {
    "single_arm": "The video shows a single-arm robot.",
    "dual_arm": "The video shows a dual-arm robot.",
    "human": "The video shows a human actor.",
    "camera_only": "The video shows the perspective of a moving camera.",
    "vehicle": "The video shows the perspective of a moving vehicle.",
}

ViewComposition = Literal[
    "separate_camera_views",
    "front",
    "above",
    "third_person",
    "wrist",
    "ego_head",
    "static_single",
    "static_single_first_person",
    "dual_wrist",
    "head_over_wrists",
    "above_over_wrists",
    "wrist_over_side_third_person",
    "above_over_side_third_person",
]
ViewCompositionMappings: dict[ViewComposition, str] = {
    "separate_camera_views": "The synchronized camera views are encoded separately, in a fixed camera order.",
    "front": "This video is captured from a dynamic front-facing perspective looking at the scene.",
    "above": "This video is captured from a third-person perspective looking towards the agent from above.",
    "third_person": "This video is captured from a third-person perspective looking towards the robot.",
    "wrist": "This video is captured from a wrist-mounted camera.",
    "ego_head": "This video is captured from a head-mounted camera showing an egocentric view of the scene.",
    "static_single": "This video is captured from a static perspective looking towards the actor.",
    "static_single_first_person": "This video is captured from a static first-person perspective looking at the scene.",
    "dual_wrist": "This video is captured from two wrist-mounted cameras.",
    "head_over_wrists": "This video contains concatenated views from multiple camera perspectives. The top row shows the head-mounted camera view looking down at the workspace. The bottom row contains two horizontally concatenated wrist-mounted camera views: the left hand camera on the left and the right hand camera on the right.",
    "above_over_wrists": "This video contains concatenated views from multiple camera perspectives. The top row shows the overhead/top camera. The bottom row contains the left camera on the left and the right camera on the right.",
    "wrist_over_side_third_person": "This video contains concatenated views from multiple camera perspectives. The top row is from the wrist-mounted camera. The bottom row contains two horizontally concatenated static third-person perspective views of the scene from opposite sides, with the robot visible.",
    "above_over_side_third_person": "This video contains concatenated views from multiple camera perspectives. The top row shows third-person static perspective view looking towards the robot from above. The bottom-left video shows the third-person static perspective view looking at the scene from the left side. The bottom-right video shows the third-person static perspective view looking at the scene from the right side.",
}

CaptionSubject = Literal[
    "camera_ego_pose",
    "right_wrist_camera",
    "dual_wrist_cameras",
    "ego_and_dual_wrist_cameras",
    "vehicle_ego_pose",
    "right_arm_end_effector",
    "right_arm_attachment_site_end_effector",
    "right_handheld_gripper_end_effector",
    "dual_arm_end_effectors",
    "dual_arm_joint_positions",
    "ego_and_dual_arm_end_effectors",
    "dual_handheld_gripper_end_effectors",
    "human_wrists_and_fingertips",
    "human_head_and_wrists",
    "human_head_wrists_and_fingertips",
]
CaptionSubjectMappings: dict[CaptionSubject, str] = {
    "right_wrist_camera": (
        "The action target is the wrist-mounted camera motion and gripper openness, "
        "mapped to the right wrist-pose and right-openness components."
    ),
    "dual_wrist_cameras": (
        "The action targets are the right and left wrist-mounted camera motions and gripper openness, "
        "mapped to the right and left wrist-pose and openness components."
    ),
    "ego_and_dual_wrist_cameras": (
        "The action targets are the head ego motion, right and left wrist-mounted camera motions, and "
        "gripper openness, mapped to the ego-pose, right and left wrist-pose and openness components."
    ),
    "dual_arm_joint_positions": (
        "The action performed by the robot is defined as the joint positions of its left and right arms "
        "and both grippers' positions."
    ),
    "camera_ego_pose": "The camera action is defined as its ego motion, mapped to the ego-pose component.",
    "vehicle_ego_pose": (
        "The action performed by the vehicle is defined as its ego motion, mapped to the ego-pose component."
    ),
    "right_arm_end_effector": (
        "The action performed by the robot is defined as the motion of its end-effector link and gripper openness, "
        "mapped to the right wrist-pose and right-openness components."
    ),
    "right_arm_attachment_site_end_effector": (
        "The action performed by the robot is defined as the motion of its attachment-site end-effector link and "
        "gripper openness, mapped to the right wrist-pose and right-openness components."
    ),
    "right_handheld_gripper_end_effector": (
        "The action demonstrated using the handheld gripper is defined as the motion of its end-effector link and "
        "gripper openness, mapped to the right wrist-pose and right-openness components."
    ),
    "dual_arm_end_effectors": (
        "The action performed by the robot is defined as the motion of its right and left end-effector links and both "
        "grippers' openness, mapped to the right and left wrist-pose and openness components."
    ),
    "ego_and_dual_arm_end_effectors": (
        "The action performed by the robot is defined as the ego motion of its head, the motion of its right and "
        "left end-effector links, and both grippers' openness, mapped to the ego-pose, right and left wrist-pose, "
        "and right and left openness components."
    ),
    "dual_handheld_gripper_end_effectors": (
        "The action demonstrated using the handheld grippers is defined as the motion of their right and left "
        "end-effector links and both grippers' openness, mapped to the right and left wrist-pose and openness "
        "components."
    ),
    "human_wrists_and_fingertips": (
        "The action performed by the human actor is defined as the wrist and fingertip motion of the actor's hands, "
        "mapped to the right and left wrist-pose and fingertip components."
    ),
    "human_head_and_wrists": (
        "The action performed by the human actor is defined as the ego pose of the actor's head and the wrist motion "
        "of the actor's hands, mapped to the ego-pose and right and left wrist-pose components."
    ),
    "human_head_wrists_and_fingertips": (
        "The action performed by the human actor is defined as the ego motion of the actor's head and the wrist and "
        "fingertip motion of the actor's hands, mapped to the ego-pose and right and left wrist-pose and fingertip "
        "components."
    ),
}

ActionCaptionAttributeValue = str | float | int
ACTION_CAPTION_POSTFIX_KEYS: tuple[str, ...] = (
    "domain_postfix",
    "embodiment_postfix",
    "view_postfix",
    "subject_postfix",
)


def append_action_caption_semantics(caption: str, attributes: Mapping[str, object]) -> str:
    """Append the registered semantic postfixes to a non-empty caption in canonical order."""
    if not caption:
        return caption

    postfixes: list[str] = []
    for key in ACTION_CAPTION_POSTFIX_KEYS:
        postfix = attributes.get(key)
        if not isinstance(postfix, str) or not postfix.strip():
            raise ValueError(f"Action caption attribute {key!r} must be a non-empty string.")
        postfixes.append(postfix.strip())

    caption = caption.rstrip()
    separator = " " if caption.endswith((".", "!", "?")) else ". "
    return f"{caption}{separator}{' '.join(postfixes)}"


@dataclass(frozen=True)
class ActionCaptionAttributeProtocol:
    """Dataset-level semantic facts that cannot be inferred from tensors."""

    domain: Domain
    embodiment: Embodiment
    view_composition: ViewComposition
    caption_subject: CaptionSubject


class ActionCaptionAttributeAdapter:
    """Resolve semantic protocol plus timing derived from one loaded window."""

    def __init__(self, protocols: Mapping[str, ActionCaptionAttributeProtocol]) -> None:
        self.protocols: dict[str, ActionCaptionAttributeProtocol] = dict(protocols)

    def supports(self, dataset_name: str) -> bool:
        """Return whether caption attributes are enabled for this dataset."""
        return dataset_name in self.protocols

    def resolve(
        self,
        dataset_name: str,
        *,
        fps: float,
        observation_count: int,
        view_count: int,
        view_composition: ViewComposition | None = None,
    ) -> dict[str, ActionCaptionAttributeValue]:
        """Return deterministic attributes for one loaded training window."""
        try:
            protocol = self.protocols[dataset_name]
        except KeyError as error:
            raise KeyError(f"No action-caption attribute protocol registered for dataset {dataset_name!r}.") from error
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}.")
        if observation_count < 1:
            raise ValueError(f"observation_count must be positive, got {observation_count}.")
        if view_count < 1:
            raise ValueError(f"view_count must be positive, got {view_count}.")
        resolved_view_composition = view_composition or protocol.view_composition
        action_transition_count = observation_count - 1
        return {
            "dataset_name": dataset_name,
            "domain": protocol.domain,
            "embodiment": protocol.embodiment,
            "view_composition": resolved_view_composition,
            "caption_subject": protocol.caption_subject,
            "domain_postfix": DomainMappings[protocol.domain],
            "embodiment_postfix": EmbodimentMappings[protocol.embodiment],
            "view_postfix": ViewCompositionMappings[resolved_view_composition],
            "subject_postfix": CaptionSubjectMappings[protocol.caption_subject],
            "fps": float(fps),
            "duration_seconds": action_transition_count / float(fps),
            "observation_count": observation_count,
            "action_transition_count": action_transition_count,
            "view_count": view_count,
        }


P = ActionCaptionAttributeProtocol

# Edit this table to update dataset semantics. Caption subjects group datasets
# with equivalent unified_v1 components; Runtime timing and counts are
# derived by ActionCaptionAttributeAdapter from the loaded training item.
ACTION_CAPTION_ATTRIBUTE_ADAPTER = ActionCaptionAttributeAdapter(
    {
        "fractal": P("real", "single_arm", "static_single_first_person", "right_arm_end_effector"),
        "bridge": P("real", "single_arm", "static_single_first_person", "right_arm_end_effector"),
        "droid": P("real", "single_arm", "wrist_over_side_third_person", "right_arm_end_effector"),
        "av": P("real", "vehicle", "front", "vehicle_ego_pose"),
        "camera_action_v2": P("real", "camera_only", "front", "camera_ego_pose"),
        "embodiment_c": P("real", "dual_arm", "head_over_wrists", "ego_and_dual_arm_end_effectors"),
        "embodiment_c_ext": P("real", "dual_arm", "head_over_wrists", "ego_and_dual_arm_end_effectors"),
        "agibot_world": P("real", "dual_arm", "head_over_wrists", "ego_and_dual_arm_end_effectors"),
        "agibot": P("real", "dual_arm", "head_over_wrists", "ego_and_dual_arm_end_effectors"),
        "embodiment_b": P("real", "dual_arm", "head_over_wrists", "ego_and_dual_arm_end_effectors"),
        "genrobot_10k": P("real", "dual_arm", "dual_wrist", "dual_handheld_gripper_end_effectors"),
        "robodojo": P("synthetic", "dual_arm", "above_over_wrists", "dual_arm_end_effectors"),
        "robodojo_joint": P("synthetic", "dual_arm", "above_over_wrists", "dual_arm_joint_positions"),
        "robodojo_real": P("real", "dual_arm", "above_over_wrists", "dual_arm_end_effectors"),
        "robodojo_real_joint": P("real", "dual_arm", "above_over_wrists", "dual_arm_joint_positions"),
        "robomind_franka": P("real", "single_arm", "above_over_side_third_person", "right_arm_end_effector"),
        "robomind_franka_dual": P("real", "dual_arm", "above_over_side_third_person", "dual_arm_end_effectors"),
        "robomind_ur": P("real", "single_arm", "above", "right_arm_attachment_site_end_effector"),
        "umi_singlearm_lerobot_480_20260625": P("real", "single_arm", "wrist", "right_handheld_gripper_end_effector"),
        "umi_singlearm_lerobot_256_20260625": P("real", "single_arm", "wrist", "right_handheld_gripper_end_effector"),
        "umi_bimanual_lerobot_480_20260625": P("real", "dual_arm", "dual_wrist", "dual_handheld_gripper_end_effectors"),
        "maniparena_tabletop": P("real", "dual_arm", "head_over_wrists", "dual_arm_end_effectors"),
        "maniparena_mobile": P("real", "dual_arm", "head_over_wrists", "ego_and_dual_arm_end_effectors"),
        "abc_130k": P("real", "dual_arm", "above_over_wrists", "dual_arm_end_effectors"),
        "molmoact2_yam": P("real", "dual_arm", "above_over_wrists", "dual_arm_end_effectors"),
        "xdof_yam_v5": P("real", "dual_arm", "above_over_wrists", "dual_arm_end_effectors"),
        "xdof_yam_annotated": P("real", "dual_arm", "above_over_wrists", "dual_arm_end_effectors"),
        "so100_midtrain_15hz": P("real", "single_arm", "third_person", "right_arm_end_effector"),
        "so101_midtrain_15hz": P("real", "single_arm", "third_person", "right_arm_end_effector"),
        "biso101_midtrain_15hz": P("real", "dual_arm", "third_person", "dual_arm_end_effectors"),
        "web_human_action_hand": P("real", "human", "static_single", "human_wrists_and_fingertips"),
        "web_human_action_body": P("real", "human", "static_single", "human_head_and_wrists"),
        "vitra_ego4d": P("real", "human", "ego_head", "human_head_wrists_and_fingertips"),
        "embodiment_a": P("real", "human", "ego_head", "human_head_wrists_and_fingertips"),
        "behavior1k_subtask": P("synthetic", "dual_arm", "head_over_wrists", "ego_and_dual_arm_end_effectors"),
    }
)

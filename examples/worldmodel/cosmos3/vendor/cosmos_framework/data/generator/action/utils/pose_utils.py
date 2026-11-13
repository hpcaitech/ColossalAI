# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Rotation and pose utilities for action datasets.

This module centralizes three related responsibilities used across the action
dataset stack:

1. Converting rotations between the conventions used by the datasets and the
   action model (`euler_xyz`, quaternion, axis-angle, rot6d, rot9d, matrix).
2. Building absolute homogeneous poses of shape ``(T, 4, 4)`` from per-frame
   translation and rotation components.
3. Converting trajectories between absolute-pose form and the relative-pose
   action vectors consumed by the datasets.

    The relative-pose action vectors always follow the shared layout
    ``[translation(3), rotation(...)]``. The rotation block is encoded with the
    requested rotation output convention, and `convert_rotation()` is the
    canonical public entrypoint for representation conversion.
"""

import math
from typing import Literal

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

PoseConvention = Literal[
    "absolute",
    "backward_anchored",
    "backward_framewise",
    "backward_chunk_anchored_8f",
    "backward_chunk_anchored_16f",
]
RotationConvention = Literal["matrix", "euler_xyz", "quat_xyzw", "quat_wxyz", "rot6d", "axisangle", "rot9d"]

_POSE_CHUNK_LENGTHS: dict[PoseConvention, int] = {
    "backward_chunk_anchored_8f": 8,
    "backward_chunk_anchored_16f": 16,
}


def _to_numpy_float32(array: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert an input array to a NumPy ``float32`` array.

    Args:
        array: A torch tensor or NumPy array with arbitrary leading dimensions.

    Returns:
        A NumPy array with dtype ``float32``. Torch tensors are moved to CPU
        before conversion. NumPy inputs are converted with ``copy=False``
        semantics when possible.

    Raises:
        ValueError: If a torch tensor with ``requires_grad=True`` is passed.
            These utilities are non-differentiable; callers must explicitly
            detach tensors before conversion.
    """
    if isinstance(array, torch.Tensor):
        if array.requires_grad:
            raise ValueError(
                "pose_utils conversion is non-differentiable; call `.detach()` "
                "explicitly before passing tensors with requires_grad=True"
            )
        return array.cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(array, dtype=np.float32)


def _normalize_rotation_matrices(rot_matrices: np.ndarray) -> np.ndarray:
    """Project approximate matrices onto valid rotation matrices.

    This helper uses an SVD-based projection onto ``SO(3)``. It is mainly used
    when decoding rotations from network-like representations such as rot6d or rot9d
    where the input may not already be perfectly orthonormal.

    Args:
        rot_matrices: Array of shape ``(..., 3, 3)`` containing one or more
            approximate rotation matrices.

    Returns:
        Array of shape ``(..., 3, 3)`` whose trailing matrices are proper
        rotation matrices with determinant ``+1``.

    Raises:
        ValueError: If the input does not have trailing shape ``(3, 3)``.
    """
    matrices = np.asarray(rot_matrices, dtype=np.float32)
    if matrices.ndim < 2 or matrices.shape[-2:] != (3, 3):
        raise ValueError(f"Rotation matrices must have shape (..., 3, 3), got {matrices.shape}")

    original_shape = matrices.shape[:-2]
    matrices_flat = matrices.reshape(-1, 3, 3)

    # Batched SVD projection to SO(3).
    U, _, Vt = np.linalg.svd(matrices_flat)
    normalized = U @ Vt

    # Ensure determinant is +1 (proper rotations, no reflections).
    det = np.linalg.det(normalized)
    reflection_mask = det < 0
    if np.any(reflection_mask):
        U_reflect = U.copy()
        U_reflect[reflection_mask, :, -1] *= -1
        normalized[reflection_mask] = U_reflect[reflection_mask] @ Vt[reflection_mask]

    return normalized.astype(np.float32, copy=False).reshape(*original_shape, 3, 3)


def convert_rotation(
    rotation: torch.Tensor | np.ndarray,
    input_format: RotationConvention,
    output_format: RotationConvention,
    normalize_matrix: bool = False,
) -> torch.Tensor | np.ndarray:
    """Convert rotations between the conventions used by action datasets.

    The function first maps the input representation to rotation matrices and
    then emits the requested output convention. It is the single conversion seam
    used by the public pose helpers so that all code paths share the same
    convention handling.

    Supported input conventions:
        - ``matrix``: rotation matrices with shape ``(..., 3, 3)``
        - ``euler_xyz``: Euler xyz angles in radians with shape ``(..., 3)``
        - ``quat_xyzw``: quaternions in SciPy's xyzw order with shape ``(..., 4)``
        - ``quat_wxyz``: quaternions in wxyz order with shape ``(..., 4)``
        - ``rot6d``: column-based 6D representation with shape ``(..., 6)``
        - ``rot9d``: flattened rotation matrices with shape ``(..., 9)``
        - ``axisangle``: axis-angle vectors with shape ``(..., 3)``

    Supported output conventions:
        - ``matrix``
        - ``euler_xyz``
        - ``quat_xyzw``
        - ``quat_wxyz``
        - ``rot6d``
        - ``axisangle``
        - ``rot9d``

    Args:
        rotation: Input rotations in the representation specified by
            ``input_format``.
        input_format: Convention used by ``rotation``.
        output_format: Convention to return.
        normalize_matrix: Whether to project intermediate matrices to a valid
            rotation before returning. This is most useful when decoding from
            approximate ``rot6d``/``rot9d`` inputs or non-unit quaternions.

    Returns:
        Rotations with the same leading shape as the input, expressed in the
        requested output convention. Torch inputs return torch outputs on the
        same device with the same dtype; NumPy inputs return NumPy arrays.

    Raises:
        ValueError: If the input shape is incompatible with ``input_format`` or
            if either format is unsupported.
    """
    input_is_tensor = isinstance(rotation, torch.Tensor)
    input_dtype = rotation.dtype if input_is_tensor else None
    input_device = rotation.device if input_is_tensor else None
    rotation_np = _to_numpy_float32(rotation)

    if input_format == "matrix":
        if rotation_np.ndim < 2 or rotation_np.shape[-2:] != (3, 3):
            raise ValueError(f"matrix rotation must have shape (..., 3, 3), got {rotation_np.shape}")
        original_shape = rotation_np.shape[:-2]
        matrices_flat = rotation_np.reshape(-1, 3, 3)
        if normalize_matrix:
            matrices_flat = _normalize_rotation_matrices(matrices_flat).reshape(-1, 3, 3)
    elif input_format == "euler_xyz":
        if rotation_np.ndim < 1 or rotation_np.shape[-1] != 3:
            raise ValueError(f"{input_format} rotation must have shape (..., 3), got {rotation_np.shape}")
        original_shape = rotation_np.shape[:-1]
        matrices_flat = R.from_euler("xyz", rotation_np.reshape(-1, 3), degrees=False).as_matrix().astype(np.float32)
    elif input_format in ("quat_xyzw", "quat_wxyz"):
        if rotation_np.ndim < 1 or rotation_np.shape[-1] != 4:
            raise ValueError(f"{input_format} rotation must have shape (..., 4), got {rotation_np.shape}")
        original_shape = rotation_np.shape[:-1]
        quaternions = rotation_np.reshape(-1, 4)
        if input_format == "quat_wxyz":
            quaternions = quaternions[:, [1, 2, 3, 0]]
        norms = np.linalg.norm(quaternions, axis=-1)
        if np.any(norms < 1e-8):
            raise ValueError(f"Found zero-norm quaternion(s) (min norm={norms.min():.2e}).")
        if normalize_matrix:
            quaternions = quaternions / norms[:, None]
        matrices_flat = R.from_quat(quaternions).as_matrix().astype(np.float32)
    elif input_format == "rot6d":
        if rotation_np.ndim < 1 or rotation_np.shape[-1] != 6:
            raise ValueError(f"{input_format} rotation must have shape (..., 6), got {rotation_np.shape}")
        original_shape = rotation_np.shape[:-1]
        rot6d_flat = rotation_np.reshape(-1, 6)
        col0 = rot6d_flat[:, :3]
        col1 = rot6d_flat[:, 3:]
        col2 = np.cross(col0, col1, axis=-1)
        matrices_flat = np.stack((col0, col1, col2), axis=-1).astype(np.float32)
        if normalize_matrix:
            matrices_flat = _normalize_rotation_matrices(matrices_flat).reshape(-1, 3, 3)
    elif input_format == "rot9d":
        if rotation_np.ndim < 1 or rotation_np.shape[-1] != 9:
            raise ValueError(f"rot9d rotation must have shape (..., 9), got {rotation_np.shape}")
        original_shape = rotation_np.shape[:-1]
        matrices_flat = rotation_np.reshape(-1, 3, 3)
        if normalize_matrix:
            matrices_flat = _normalize_rotation_matrices(matrices_flat).reshape(-1, 3, 3)
    elif input_format == "axisangle":
        if rotation_np.ndim < 1 or rotation_np.shape[-1] != 3:
            raise ValueError(f"axisangle rotation must have shape (..., 3), got {rotation_np.shape}")
        original_shape = rotation_np.shape[:-1]
        matrices_flat = R.from_rotvec(rotation_np.reshape(-1, 3)).as_matrix().astype(np.float32)
    else:
        raise ValueError(f"Unsupported input_format: {input_format!r}")

    if output_format == "matrix":
        converted = matrices_flat.reshape(*original_shape, 3, 3).astype(np.float32)
    elif output_format == "rot9d":
        converted = matrices_flat.reshape(-1, 9)
    elif output_format == "rot6d":
        converted = matrices_flat[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    elif output_format == "quat_xyzw":
        converted = R.from_matrix(matrices_flat).as_quat().astype(np.float32)
    elif output_format == "quat_wxyz":
        converted = R.from_matrix(matrices_flat).as_quat().astype(np.float32)
        converted = converted[:, [3, 0, 1, 2]]
    elif output_format == "euler_xyz":
        converted = R.from_matrix(matrices_flat).as_euler("xyz", degrees=False).astype(np.float32)
    elif output_format == "axisangle":
        converted = R.from_matrix(matrices_flat).as_rotvec().astype(np.float32)
    else:
        raise ValueError(f"Unsupported output_format: {output_format!r}")

    if output_format != "matrix":
        converted = converted.reshape(*original_shape, converted.shape[-1])

    if input_is_tensor:
        return torch.from_numpy(np.ascontiguousarray(converted)).to(dtype=input_dtype, device=input_device)
    return converted


# -----------------------------------------------------------------------------
# Absolute pose construction
# -----------------------------------------------------------------------------


def build_abs_pose_from_components(
    xyz: torch.Tensor | np.ndarray,
    rotation: torch.Tensor | np.ndarray,
    rotation_input_format: Literal["euler_xyz", "quat_xyzw", "quat_wxyz", "axisangle"],
    translation_scale: float | None = None,
) -> np.ndarray:
    """Build absolute homogeneous poses from per-frame translation and rotation.

    This is the canonical helper for turning dataset-provided pose components
    into a sequence of rigid transforms. Each output pose is a homogeneous
    transform whose top-left ``3 x 3`` block stores rotation and whose last
    column stores translation.

    Args:
        xyz: Per-frame translations with shape ``(T, 3)``.
        rotation: Per-frame rotations with shape ``(T, 3)`` for ``euler_xyz``
            and ``axisangle``, or ``(T, 4)`` for quaternion conventions.
        rotation_input_format: Convention used by ``rotation``. Supported values
            are ``euler_xyz``, ``quat_xyzw``, ``quat_wxyz``, and ``axisangle``.
        translation_scale: Optional factor used to divide translations before
            inserting them into the output poses. This is useful when upstream
            data stores translations in a scaled unit.

    Returns:
        Absolute poses with shape ``(T, 4, 4)`` and dtype ``float32``.

    Raises:
        ValueError: If the translation and rotation arrays have incompatible
            lengths or unsupported shapes, or if ``translation_scale`` is zero.
    """
    xyz_np = _to_numpy_float32(xyz)
    rotation_np = _to_numpy_float32(rotation)

    if xyz_np.ndim != 2 or xyz_np.shape[1] != 3:
        raise ValueError(f"xyz must have shape (T, 3), got {xyz_np.shape}")
    if rotation_np.ndim != 2:
        raise ValueError(f"rotation must be 2D, got {rotation_np.shape}")
    if rotation_np.shape[0] != xyz_np.shape[0]:
        raise ValueError(
            f"xyz and rotation must have the same length, got {xyz_np.shape[0]} and {rotation_np.shape[0]}"
        )

    rot_mats = np.asarray(
        convert_rotation(rotation_np, input_format=rotation_input_format, output_format="matrix"),
        dtype=np.float32,
    )

    if translation_scale is not None:
        if translation_scale == 0:
            raise ValueError("translation_scale must be non-zero")
        xyz_np = xyz_np / float(translation_scale)

    poses_abs = np.eye(4, dtype=np.float32)[None].repeat(xyz_np.shape[0], axis=0)
    poses_abs[:, :3, :3] = rot_mats.astype(np.float32)
    poses_abs[:, :3, 3] = xyz_np
    return poses_abs


# -----------------------------------------------------------------------------
# Relative pose conversions
# -----------------------------------------------------------------------------


def _delta_transform_to_pose_vector(
    delta_T: np.ndarray,
    rotation_output_format: RotationConvention,
) -> np.ndarray:
    """Encode a relative transform as an action vector.

    The shared action-vector layout is always ``[translation(3), rotation(...)]``.

    Args:
        delta_T: Relative transform of shape ``(4, 4)``.
        rotation_output_format: Concrete convention used for the output rotation
            block.

    Returns:
        A ``float32`` action vector whose first three values are translation and
        whose remaining values are the rotation in ``rotation_output_format``.
    """
    delta_np = np.asarray(delta_T, dtype=np.float32)
    if delta_np.shape != (4, 4):
        raise ValueError(f"delta_T must have shape (4, 4), got {delta_np.shape}")

    translation = delta_np[:3, 3]
    rotation = np.asarray(
        convert_rotation(delta_np[:3, :3], input_format="matrix", output_format=rotation_output_format),
        dtype=np.float32,
    )
    return np.concatenate([translation, rotation]).astype(np.float32)


def _pose_vector_to_delta_transform(
    pose_vector: np.ndarray,
    rotation_input_format: RotationConvention,
    translation_scale: float,
    normalize_rotation: bool,
    rotation_scale: float = 1.0,
) -> np.ndarray:
    """Decode an action vector back into a relative homogeneous transform.

    This is the inverse of `_delta_transform_to_pose_vector()` when the same
    rotation convention is used. Scale arguments are provided for callers that
    need to decode model-space pose actions before action-normalizer
    denormalization has been applied.

    Args:
        pose_vector: Relative-pose action vector with layout
            ``[translation(3), rotation(...)]``.
        rotation_input_format: Concrete convention used by the rotation block.
        translation_scale: Scalar used to undo translation scaling in the input
            vector.
        normalize_rotation: Whether to project the decoded rotation to a valid
            matrix before assembling the transform.
        rotation_scale: Scalar used to undo rotation scaling in the input vector.

    Returns:
        A relative homogeneous transform with shape ``(4, 4)`` and dtype
        ``float32``.
    """
    pose_vector_np = np.asarray(pose_vector, dtype=np.float32)
    rotation_block = pose_vector_np[3:] / rotation_scale

    rotation_matrix = np.asarray(
        convert_rotation(
            rotation_block,
            input_format=rotation_input_format,
            output_format="matrix",
            normalize_matrix=normalize_rotation,
        ),
        dtype=np.float32,
    )

    delta_T = np.eye(4, dtype=np.float32)
    delta_T[:3, 3] = pose_vector_np[:3] / translation_scale
    delta_T[:3, :3] = rotation_matrix
    return delta_T


def _get_relative_delta_transform(
    poses_abs: np.ndarray,
    inv_poses_abs: np.ndarray,
    frame_idx: int,
    pose_convention: PoseConvention,
    anchor_index: int = 0,
) -> np.ndarray:
    """Compute one relative transform from an absolute-pose trajectory.

    Args:
        poses_abs: Absolute poses of shape ``(T, 4, 4)``.
        inv_poses_abs: Precomputed inverses of ``poses_abs`` with the same shape.
        frame_idx: Index of the output row to encode, in ``[0, T - 2]``.
        pose_convention: Pose convention controlling which two poses
            define the delta and whether it is framewise or anchored.
        anchor_index: For ``backward_anchored`` only: index of the anchor pose.
            Row ``frame_idx`` encodes the frame at ``frame_idx`` when it precedes
            the anchor and ``frame_idx + 1`` otherwise, so the anchor frame itself
            (which would encode as the identity) is skipped and the output stays
            ``T - 1`` rows for any anchor. ``backward_framewise`` has no anchor and
            requires the default ``0``.

    Returns:
        The relative transform ``delta_T`` with shape ``(4, 4)`` for the
        requested step and convention.
    """
    if pose_convention == "backward_framewise":
        if anchor_index != 0:
            raise ValueError(f"anchor_index is only meaningful for backward_anchored, got {anchor_index} (framewise).")
        return inv_poses_abs[frame_idx] @ poses_abs[frame_idx + 1]
    if pose_convention == "backward_anchored":
        # `anchor_index` generalises a fixed anchor of 0: at anchor_index=0 this reduces
        # exactly to `inv_poses_abs[0] @ poses_abs[frame_idx + 1]`. The generalisation is
        # what makes anchored encoding usable with action history, where the anchor is the
        # current frame rather than the window start.
        target = frame_idx if frame_idx < anchor_index else frame_idx + 1
        return inv_poses_abs[anchor_index] @ poses_abs[target]
    if chunk_length := _POSE_CHUNK_LENGTHS.get(pose_convention):
        # The chunk-anchored conventions deliberately do NOT use the anchor-skip above: their
        # target is always >= anchor_idx + 1, so no identity row can arise and no skip is needed.
        if anchor_index != 0:
            raise ValueError(f"anchor_index is only meaningful for backward_anchored, got {anchor_index}.")
        anchor_idx = (frame_idx // chunk_length) * chunk_length
        return inv_poses_abs[anchor_idx] @ poses_abs[frame_idx + 1]
    raise ValueError(
        f"Unsupported pose_convention={pose_convention!r}. Expected one of: backward_framewise, "
        f"backward_anchored, {', '.join(sorted(_POSE_CHUNK_LENGTHS))}."
    )


def _apply_relative_delta_transform(
    current_pose: np.ndarray,
    initial_pose: np.ndarray,
    chunk_anchor: np.ndarray,
    delta_T: np.ndarray,
    pose_convention: PoseConvention,
) -> np.ndarray:
    """Recover the next absolute pose from a decoded relative transform.

    Args:
        current_pose: The current reconstructed pose for framewise modes.
        initial_pose: The anchor pose used by anchored modes.
        chunk_anchor: Fixed anchor for the current generated chunk.
        delta_T: Relative transform for the current step.
        pose_convention: Pose convention controlling how ``delta_T``
            should be composed back into an absolute pose.

    Returns:
        The next absolute pose with shape ``(4, 4)``.
    """
    if pose_convention == "backward_framewise":
        return current_pose @ delta_T
    if pose_convention == "backward_anchored":
        return initial_pose @ delta_T
    if pose_convention in _POSE_CHUNK_LENGTHS:
        return chunk_anchor @ delta_T
    raise ValueError(f"Unsupported pose_convention={pose_convention!r}")


def pose_abs_to_rel(
    poses_abs: np.ndarray,
    rotation_format: RotationConvention = "rot9d",
    pose_convention: PoseConvention = "backward_framewise",
    anchor_index: int = 0,
) -> np.ndarray:
    """Convert an absolute-pose trajectory into relative-pose action vectors.

    Args:
        poses_abs: Absolute poses with shape ``(T, 4, 4)``. These are typically
            object-in-world or camera-to-world transforms.
        rotation_format: Rotation convention used for the output rotation block.
            Supported values are ``rot9d``, ``rot6d``, ``quat_xyzw``, and
            ``euler_xyz``.
        pose_convention: Pose convention:
            - ``backward_framewise``: ``delta_T = T_i^{-1} @ T_{i+1}``
            - ``backward_anchored``: ``delta_T = T_a^{-1} @ T_k`` for every frame
              ``k != a``, where ``a = anchor_index``. With ``anchor_index = 0``
              this is ``T_0^{-1} @ T_{i+1}``.
            - ``backward_chunk_anchored_8f`` and ``backward_chunk_anchored_16f``:
              each group of 8 or 16 targets is relative to the pose immediately
              before the group.
        anchor_index: For ``backward_anchored``: which frame anchors the window.
            With action history the anchor is the *current* frame (index ``H``),
            so the ``H`` rows before it point backward (where the EEF was,
            relative to now) and the rows after it point forward (where it should
            go). The anchor frame itself is skipped -- it would encode as the
            identity -- keeping the output at ``T - 1`` rows. The default ``0``
            reproduces the history-free behavior (anchor at the window start,
            which *is* the current frame when ``H = 0``). Must be ``0`` for
            ``backward_framewise`` and for the chunk-anchored conventions.

    Returns:
        An array of shape ``(T - 1, D)`` where ``D = 3 + rotation_dim``.

    Raises:
        AssertionError: If fewer than two absolute poses are provided.
        ValueError: If ``anchor_index`` is out of range for ``poses_abs``.
    """
    num_frames = len(poses_abs)
    assert num_frames > 1, "At least 2 frames are required to compute relative poses"
    if not 0 <= anchor_index < num_frames:
        raise ValueError(f"anchor_index={anchor_index} out of range for {num_frames} poses.")

    # Compute inverse poses
    inv_poses_abs = np.linalg.inv(poses_abs)

    poses_rel = []
    # We produce num_frames - 1 relative poses
    for i in range(num_frames - 1):
        delta_T = _get_relative_delta_transform(poses_abs, inv_poses_abs, i, pose_convention, anchor_index)
        poses_rel.append(
            _delta_transform_to_pose_vector(
                delta_T,
                rotation_output_format=rotation_format,
            )
        )

    return np.stack(poses_rel).astype(np.float32)  # [T-1,D]


def pose_rel_to_abs(
    poses_rel: np.ndarray,
    rotation_format: RotationConvention = "rot9d",
    pose_convention: PoseConvention = "backward_framewise",
    initial_pose: np.ndarray | None = None,
    normalize_rotation: bool = True,
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
    anchor_index: int = 0,
) -> np.ndarray:
    """Reconstruct an absolute-pose trajectory from relative-pose action vectors.

    Args:
        poses_rel: Relative-pose action vectors with shape ``(T - 1, D)`` and
            layout ``[translation(3), rotation(...)]``.
        rotation_format: Convention used by the rotation block of ``poses_rel``.
        pose_convention: Pose convention used when the vectors were
            encoded. This must match the convention passed to `pose_abs_to_rel()`.
        initial_pose: Absolute pose for the anchor frame (``backward_anchored``)
            or the first frame (``backward_framewise``). If ``None``, the
            identity transform is used.
        normalize_rotation: Whether to project decoded rotations onto ``SO(3)``
            before composing them back into the trajectory.
        translation_scale: Scalar used to undo translation scaling in
            ``poses_rel``. Prefer denormalizing with the dataset action
            normalizer before calling this function.
        rotation_scale: Scalar used to undo rotation scaling in ``poses_rel``.
            Prefer denormalizing with the dataset action normalizer before
            calling this function.
        anchor_index: Must match the value passed to :func:`pose_abs_to_rel`.
            For ``backward_anchored``, ``initial_pose`` is placed at this frame
            index in the output and every row decodes independently as
            ``initial_pose @ delta`` (rows before the anchor are past frames,
            rows after it future frames). Must be ``0`` for
            ``backward_framewise``.

    Returns:
        Absolute poses with shape ``(T, 4, 4)`` where ``T = len(poses_rel) + 1``.
    """
    if initial_pose is None:
        initial_pose = np.eye(4)

    num_poses_rel = poses_rel.shape[0]
    if not 0 <= anchor_index <= num_poses_rel:
        raise ValueError(f"anchor_index={anchor_index} out of range for {num_poses_rel} relative poses.")
    if pose_convention != "backward_anchored" and anchor_index != 0:
        raise ValueError(f"anchor_index is only meaningful for backward_anchored, got {anchor_index}.")

    if pose_convention == "backward_anchored" and anchor_index != 0:
        # For a mid-window anchor, rows decode independently and the anchor pose must be
        # inserted at its frame position to recover the original trajectory order.
        poses_abs = [
            initial_pose
            @ _pose_vector_to_delta_transform(
                pose_rel,
                rotation_input_format=rotation_format,
                translation_scale=translation_scale,
                normalize_rotation=normalize_rotation,
                rotation_scale=rotation_scale,
            )
            for pose_rel in poses_rel
        ]
        poses_abs.insert(anchor_index, initial_pose)
        return np.stack(poses_abs)  # [T,4,4]

    # Preserve the existing reconstruction path when the anchor remains at frame 0.
    poses_abs = [initial_pose]
    current_pose = initial_pose
    chunk_anchor = initial_pose
    chunk_length = _POSE_CHUNK_LENGTHS.get(pose_convention)

    for i in range(num_poses_rel):
        if chunk_length is not None and i > 0 and i % chunk_length == 0:
            chunk_anchor = current_pose
        delta_T = _pose_vector_to_delta_transform(
            poses_rel[i],
            rotation_input_format=rotation_format,
            translation_scale=translation_scale,
            normalize_rotation=normalize_rotation,
            rotation_scale=rotation_scale,
        )
        next_pose = _apply_relative_delta_transform(
            current_pose,
            initial_pose,
            chunk_anchor,
            delta_T,
            pose_convention,
        )

        poses_abs.append(next_pose)
        current_pose = next_pose

    return np.stack(poses_abs)  # [T,4,4]


def pose_anchored_to_framewise(
    poses_rel: np.ndarray,
    rotation_format: RotationConvention = "rot6d",
) -> np.ndarray:
    """Re-encode anchored relative-pose rows as framewise one-step rows.

    Idleness (and any other per-step property) is a fact about the underlying
    trajectory, not its encoding: anchored rows ``P_k = T_anchor^-1 @ T_k`` are
    losslessly convertible to one-step deltas ``delta_k = P_{k-1}^-1 @ P_k``
    (with ``P_0 = I``, since the anchor precedes the first row). This enables
    per-step checks written for the framewise convention -- e.g.
    :func:`compute_idle_frames` -- to run on anchored action chunks.

    Only forward-of-anchor rows may be passed (the dataset's ``action`` chunk;
    ``anchor_index`` 0 semantics). History rows would need their own ordering.

    Args:
        poses_rel: ``(K, D)`` anchored rows, layout ``[translation(3), rotation]``.
        rotation_format: Rotation convention of the rows' rotation block.

    Returns:
        ``(K, D)`` framewise rows in the same layout and rotation convention.
    """
    poses_abs = pose_rel_to_abs(poses_rel, rotation_format=rotation_format, pose_convention="backward_anchored")
    return pose_abs_to_rel(poses_abs, rotation_format=rotation_format, pose_convention="backward_framewise")


# -----------------------------------------------------------------------------
# Absolute 9D pose vectors: [pos(3), rot6d(6)]
# -----------------------------------------------------------------------------


def pose_abs_to_abs9d(poses_abs: np.ndarray) -> np.ndarray:
    """Convert absolute poses to absolute ``[pos(3), rot6d(6)]`` vectors.

    Both halves of the name mean different things: the input is an absolute
    ``(T, 4, 4)`` trajectory, and the output keeps absolute values rather than
    per-step deltas.

    The leading frame is dropped so the output aligns step-for-step with
    ``pose_abs_to_rel(..., rotation_format="rot6d")``, which spends one frame
    forming its first delta.

    Args:
        poses_abs: Absolute poses with shape ``(T, 4, 4)``.

    Returns:
        Array of shape ``(T - 1, 9)`` and dtype ``float32``.
    """
    pos = poses_abs[1:, :3, 3].astype(np.float32)  # (T-1, 3)
    rot6d = convert_rotation(poses_abs[1:, :3, :3], input_format="matrix", output_format="rot6d")  # (T-1, 6)
    return np.concatenate([pos, rot6d], axis=-1)


# -----------------------------------------------------------------------------
# Idle-frame detection
# -----------------------------------------------------------------------------


def _identity_rotation_vector(rotation_format: RotationConvention) -> np.ndarray:
    """Return the identity-rotation vector for a given rotation convention.

    Used by :func:`compute_idle_frames` to test whether a rotation block is
    close to "no rotation" in its current encoding.
    """
    if rotation_format in ("matrix", "rot9d"):
        return np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)
    if rotation_format == "rot6d":
        return np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    if rotation_format == "quat_xyzw":
        return np.array([0, 0, 0, 1], dtype=np.float32)
    if rotation_format == "quat_wxyz":
        return np.array([1, 0, 0, 0], dtype=np.float32)
    if rotation_format in ("euler_xyz", "axisangle"):
        return np.array([0, 0, 0], dtype=np.float32)
    raise ValueError(f"Unsupported rotation_format={rotation_format!r}")


def _rotation_angle_per_arm(rotations: np.ndarray, rotation_format: str) -> np.ndarray:
    """Geodesic angle (rad) from identity for each arm at each frame.

    ``rotations`` has shape ``(T, n_arms, n_per_arm)``; the returned array has
    shape ``(T, n_arms)``. The angle is rotation-format aware so a fixed
    ``eps_r`` threshold has consistent geometric meaning across formats:

    - ``rot6d``  → reconstruct ``trace(R)`` in closed form from the two stored
      columns ``a, b`` (already unit-orthogonal as they came from a valid
      rotation matrix). The third column is ``a × b``, so
      ``trace(R) = a[0] + b[1] + a[0]·b[1] - a[1]·b[0]``.
      ``angle = arccos(clip((trace - 1) / 2, -1, 1))``.
    - ``rot9d``  → reshape to ``(..., 3, 3)`` and use
      ``trace(R) = R[0,0] + R[1,1] + R[2,2]``.
    - ``quat_xyzw`` / ``quat_wxyz`` → ``angle = 2 · arccos(|q_w|)``; the
      absolute value handles the double cover (``q`` and ``-q`` represent the
      same rotation).
    - ``axisangle`` → the magnitude of the axis-angle vector *is* the angle.
    - ``euler_xyz`` → no closed-form angle; use ``‖euler‖`` as a conservative
      upper bound (exact for single-axis rotations, an overestimate for
      composed ones — fine for idle detection where small angles are the
      regime of interest).
    """
    if rotation_format == "rot6d":
        a = rotations[..., :3]
        b = rotations[..., 3:6]
        trace = a[..., 0] + b[..., 1] + a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        return np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    if rotation_format == "rot9d":
        mat = rotations.reshape(*rotations.shape[:-1], 3, 3)
        trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
        return np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    if rotation_format in ("quat_xyzw", "quat_wxyz"):
        qw = rotations[..., 3] if rotation_format == "quat_xyzw" else rotations[..., 0]
        return 2.0 * np.arccos(np.clip(np.abs(qw), 0.0, 1.0))
    if rotation_format == "axisangle":
        return np.linalg.norm(rotations, axis=-1)
    if rotation_format == "euler_xyz":
        # Exact for single-axis rotations, overestimate for composed ones —
        # safe for idle thresholds since overestimation can only mark a frame
        # as non-idle, never spuriously idle.
        return np.linalg.norm(rotations, axis=-1)
    raise ValueError(f"Unsupported rotation_format={rotation_format!r}")


def _consecutive_streaks(idle: np.ndarray, min_streak: int) -> np.ndarray:
    """Zero out idle bits not belonging to a run of ``>= min_streak`` Trues.

    Pure-numpy two-pointer scan. ``min_streak <= 1`` is a no-op (returns the
    input mask unchanged).
    """
    if min_streak <= 1:
        return idle
    out = np.zeros_like(idle)
    n = len(idle)
    i = 0
    while i < n:
        if not idle[i]:
            i += 1
            continue
        j = i
        while j < n and idle[j]:
            j += 1
        if j - i >= min_streak:
            out[i:j] = True
        i = j
    return out


def compute_idle_frames(
    action_raw: torch.Tensor | np.ndarray,
    spec: "ActionSpec",  # noqa: F821 — forward ref, real import is in action_spec.py
    *,
    eps_t: float = 1e-3,
    eps_r: float = math.radians(5.0),
    eps_g: float = 1e-2,
    joint_threshold: float = 5e-4,
    min_streak: int = 3,
) -> int:
    """Count idle frames in a raw (un-normalized) action chunk.

    Idle detection runs per-DimType (driven by ``spec.types``); a frame is
    *raw-idle* iff every relevant type group is idle on that frame, and
    counts toward the final tally only if it belongs to a run of at least
    ``min_streak`` consecutive raw-idle frames. The streak filter rejects
    isolated low-motion frames (instantaneous slowdowns) which carry weak
    physical meaning and add noise to the IdleFrames training signal.

    DimType branches:

    - ``POS``      → combined ``‖action[pos_idx]‖`` (L2 across all POS dims)
      < ``eps_t``. For single-arm specs (3 dims) this is the standard ``‖t‖``
      check; for multi-arm specs the combined norm is slightly stricter than
      a per-arm check.
    - ``ROT``      → per-arm geodesic rotation angle (rad) from identity
      < ``eps_r``. The angle is computed in a rotation-format aware way (see
      :func:`_rotation_angle_per_arm`) so the threshold has consistent
      geometric meaning regardless of the encoding.
    - ``GRIPPER``  → ``max |action[t] - action[t-1]| < eps_g``. ``np.diff``
      with ``prepend=action[0]`` makes step 0 ``|0|`` (treated as "no change");
      with the streak filter this can no longer create a spurious single-frame
      idle event.
    - ``JOINT``    → same frame-diff scheme as gripper with
      ``joint_threshold`` (rad / step).
    - ``RESERVED`` → ignored.

    Defaults (in the units of the un-normalized action):

    - ``eps_t = 1e-3``     → 1 mm per-frame translation
    - ``eps_r = 5°``       → 5° per-frame rotation (geodesic angle)
    - ``eps_g = 1e-2``     → 1 % gripper command change
    - ``joint_threshold = 5e-4`` → ~0.03° / step joint angle change
    - ``min_streak = 3``   → require a run of >= 3 consecutive idle frames

    The input must be **un-normalized** so the identity transform sits at
    known coordinates (translation ≈ 0, rotation ≈ identity). The action
    vector is also assumed to be encoded in a per-step / framewise convention
    (e.g. ``backward_framewise``); anchored conventions (``backward_anchored``)
    accumulate over the chunk and would silently break the POS/ROT idle
    checks. Callers (e.g. the LeRobot base class) gate on pose convention
    before calling this function.
    """
    if isinstance(action_raw, torch.Tensor):
        action = action_raw.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        action = np.asarray(action_raw, dtype=np.float32)

    if action.ndim != 2:
        raise ValueError(f"action_raw must be 2-D (T, D); got shape {action.shape}")
    num_frames, action_dim = action.shape
    if num_frames == 0:
        return 0
    if action_dim != len(spec.types):
        raise ValueError(f"action_dim={action_dim} does not match spec.dim={len(spec.types)}")

    # Import locally to avoid a circular import at module load time
    # (action_spec.py imports RotationConvention from this file).
    from cosmos_framework.data.generator.action.utils.action_spec import DimType

    pos_idx = [i for i, t in enumerate(spec.types) if t == DimType.POS]
    rot_idx = [i for i, t in enumerate(spec.types) if t == DimType.ROT]
    grip_idx = [i for i, t in enumerate(spec.types) if t == DimType.GRIPPER]
    joint_idx = [i for i, t in enumerate(spec.types) if t == DimType.JOINT]

    idle = np.ones(num_frames, dtype=bool)

    # POS: combined L2 norm across all translation dims.
    if pos_idx:
        idle &= np.linalg.norm(action[:, pos_idx], axis=1) < eps_t

    # ROT: per-arm geodesic angle (rad).
    if rot_idx:
        rot_id = _identity_rotation_vector(spec.rotation_format)
        n_per_arm = rot_id.shape[0]
        if len(rot_idx) % n_per_arm != 0:
            raise ValueError(
                f"ROT dims ({len(rot_idx)}) not a multiple of "
                f"rotation_format={spec.rotation_format!r} dim ({n_per_arm})"
            )
        rotations = action[:, rot_idx].reshape(num_frames, -1, n_per_arm)
        angles = _rotation_angle_per_arm(rotations, spec.rotation_format)  # (T, n_arms)
        idle &= angles.max(axis=1) < eps_r

    # GRIPPER: max |Δgripper| across all gripper dims; step 0's diff is 0.
    if grip_idx:
        gripper = action[:, grip_idx]
        diff = np.abs(np.diff(gripper, axis=0, prepend=gripper[:1]))
        idle &= diff.max(axis=1) < eps_g

    # JOINT: same frame-diff scheme with joint_threshold.
    if joint_idx:
        joints = action[:, joint_idx]
        diff = np.abs(np.diff(joints, axis=0, prepend=joints[:1]))
        idle &= diff.max(axis=1) < joint_threshold

    if min_streak > 1:
        idle = _consecutive_streaks(idle, min_streak)

    return int(idle.sum())


def compute_framewise_idle_frames(
    action_raw: torch.Tensor | np.ndarray,
    spec: "ActionSpec",  # noqa: F821 — forward ref, real import is in action_spec.py
    *,
    fps: float,
    pose_convention: PoseConvention,
    eps_t_per_sec: float = 5e-3,
    eps_r_per_sec: float = math.radians(1.5),
    eps_g: float = 1e-2,
    joint_threshold_per_sec: float = 5e-3,
    min_streak: int = 3,
    eps_t: float | None = None,
    eps_r: float | None = None,
    joint_threshold: float | None = None,
) -> torch.Tensor | None:  # [] or None
    """Apply the shared framewise idle policy and return a trainer scalar.

    Velocity thresholds are converted from per-second to per-frame units using
    ``fps`` unless an explicit per-frame override is provided. Anchored and
    absolute pose conventions do not have framewise idle semantics.
    """
    if pose_convention != "backward_framewise":
        return None
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")

    idle_frame_count = compute_idle_frames(
        action_raw,
        spec,
        eps_t=eps_t if eps_t is not None else eps_t_per_sec / fps,
        eps_r=eps_r if eps_r is not None else eps_r_per_sec / fps,
        eps_g=eps_g,
        joint_threshold=(joint_threshold if joint_threshold is not None else joint_threshold_per_sec / fps),
        min_streak=min_streak,
    )
    return torch.tensor(idle_frame_count, dtype=torch.long)  # []

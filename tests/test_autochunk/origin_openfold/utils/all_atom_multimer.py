# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ops for all atom representations."""

from functools import partial
from typing import Dict, Text, Tuple

import numpy as np
import torch
from fastfold.common import residue_constants as rc
from fastfold.utils import geometry, tensor_utils


def squared_difference(x, y):
    return np.square(x - y)


def get_rc_tensor(rc_np, aatype):
    return torch.tensor(rc_np, device=aatype.device)[aatype]


def atom14_to_atom37(
        atom14_data: torch.Tensor,    # (*, N, 14, ...)
        aatype: torch.Tensor    # (*, N)
) -> torch.Tensor:    # (*, N, 37, ...)
    """Convert atom14 to atom37 representation."""
    idx_atom37_to_atom14 = get_rc_tensor(rc.RESTYPE_ATOM37_TO_ATOM14, aatype)
    no_batch_dims = len(aatype.shape) - 1
    atom37_data = tensor_utils.batched_gather(atom14_data,
                                              idx_atom37_to_atom14,
                                              dim=no_batch_dims + 1,
                                              no_batch_dims=no_batch_dims + 1)
    atom37_mask = get_rc_tensor(rc.RESTYPE_ATOM37_MASK, aatype)
    if len(atom14_data.shape) == no_batch_dims + 2:
        atom37_data *= atom37_mask
    elif len(atom14_data.shape) == no_batch_dims + 3:
        atom37_data *= atom37_mask[..., None].astype(atom37_data.dtype)
    else:
        raise ValueError("Incorrectly shaped data")
    return atom37_data


def atom37_to_atom14(aatype, all_atom_pos, all_atom_mask):
    """Convert Atom37 positions to Atom14 positions."""
    residx_atom14_to_atom37 = get_rc_tensor(rc.RESTYPE_ATOM14_TO_ATOM37, aatype)
    no_batch_dims = len(aatype.shape)
    atom14_mask = tensor_utils.batched_gather(
        all_atom_mask,
        residx_atom14_to_atom37,
        dim=no_batch_dims + 1,
        no_batch_dims=no_batch_dims + 1,
    ).to(torch.float32)
    # create a mask for known groundtruth positions
    atom14_mask *= get_rc_tensor(rc.RESTYPE_ATOM14_MASK, aatype)
    # gather the groundtruth positions
    atom14_positions = tensor_utils.batched_gather(
        all_atom_pos,
        residx_atom14_to_atom37,
        dim=no_batch_dims + 1,
        no_batch_dims=no_batch_dims + 1,
    ),
    atom14_positions = atom14_mask * atom14_positions
    return atom14_positions, atom14_mask


def get_alt_atom14(aatype, positions: torch.Tensor, mask):
    """Get alternative atom14 positions."""
    # pick the transformation matrices for the given residue sequence
    # shape (num_res, 14, 14)
    renaming_transform = get_rc_tensor(rc.RENAMING_MATRICES, aatype)
    alternative_positions = torch.sum(positions[..., None, :] * renaming_transform[..., None], dim=-2)

    # Create the mask for the alternative ground truth (differs from the
    # ground truth mask, if only one of the atoms in an ambiguous pair has a
    # ground truth position)
    alternative_mask = torch.sum(mask[..., None] * renaming_transform, dim=-2)

    return alternative_positions, alternative_mask


def atom37_to_frames(
        aatype: torch.Tensor,    # (...)
        all_atom_positions: torch.Tensor,    # (..., 37)
        all_atom_mask: torch.Tensor,    # (..., 37)
) -> Dict[Text, torch.Tensor]:
    """Computes the frames for the up to 8 rigid groups for each residue."""
    # 0: 'backbone group',
    # 1: 'pre-omega-group', (empty)
    # 2: 'phi-group', (currently empty, because it defines only hydrogens)
    # 3: 'psi-group',
    # 4,5,6,7: 'chi1,2,3,4-group'

    no_batch_dims = len(aatype.shape) - 1

    # Compute the gather indices for all residues in the chain.
    # shape (N, 8, 3)
    residx_rigidgroup_base_atom37_idx = get_rc_tensor(rc.RESTYPE_RIGIDGROUP_BASE_ATOM37_IDX, aatype)

    # Gather the base atom positions for each rigid group.
    base_atom_pos = tensor_utils.batched_gather(
        all_atom_positions,
        residx_rigidgroup_base_atom37_idx,
        dim=no_batch_dims + 1,
        batch_dims=no_batch_dims + 1,
    )

    # Compute the Rigids.
    point_on_neg_x_axis = base_atom_pos[..., :, :, 0]
    origin = base_atom_pos[..., :, :, 1]
    point_on_xy_plane = base_atom_pos[..., :, :, 2]
    gt_rotation = geometry.Rot3Array.from_two_vectors(origin - point_on_neg_x_axis, point_on_xy_plane - origin)

    gt_frames = geometry.Rigid3Array(gt_rotation, origin)

    # Compute a mask whether the group exists.
    # (N, 8)
    group_exists = get_rc_tensor(rc.RESTYPE_RIGIDGROUP_MASK, aatype)

    # Compute a mask whether ground truth exists for the group
    gt_atoms_exist = tensor_utils.batched_gather(    # shape (N, 8, 3)
        all_atom_mask.to(dtype=torch.float32),
        residx_rigidgroup_base_atom37_idx,
        batch_dims=no_batch_dims + 1,
    )
    gt_exists = torch.min(gt_atoms_exist, dim=-1) * group_exists    # (N, 8)

    # Adapt backbone frame to old convention (mirror x-axis and z-axis).
    rots = np.tile(np.eye(3, dtype=np.float32), [8, 1, 1])
    rots[0, 0, 0] = -1
    rots[0, 2, 2] = -1
    gt_frames = gt_frames.compose_rotation(geometry.Rot3Array.from_array(torch.tensor(rots, device=aatype.device)))

    # The frames for ambiguous rigid groups are just rotated by 180 degree around
    # the x-axis. The ambiguous group is always the last chi-group.
    restype_rigidgroup_is_ambiguous = np.zeros([21, 8], dtype=np.float32)
    restype_rigidgroup_rots = np.tile(np.eye(3, dtype=np.float32), [21, 8, 1, 1])

    for resname, _ in rc.residue_atom_renaming_swaps.items():
        restype = rc.restype_order[rc.restype_3to1[resname]]
        chi_idx = int(sum(rc.chi_angles_mask[restype]) - 1)
        restype_rigidgroup_is_ambiguous[restype, chi_idx + 4] = 1
        restype_rigidgroup_rots[restype, chi_idx + 4, 1, 1] = -1
        restype_rigidgroup_rots[restype, chi_idx + 4, 2, 2] = -1

    # Gather the ambiguity information for each residue.
    residx_rigidgroup_is_ambiguous = torch.tensor(
        restype_rigidgroup_is_ambiguous,
        device=aatype.device,
    )[aatype]
    ambiguity_rot = torch.tensor(
        restype_rigidgroup_rots,
        device=aatype.device,
    )[aatype]
    ambiguity_rot = geometry.Rot3Array.from_array(torch.Tensor(ambiguity_rot, device=aatype.device))

    # Create the alternative ground truth frames.
    alt_gt_frames = gt_frames.compose_rotation(ambiguity_rot)

    fix_shape = lambda x: x.reshape(x.shape[:-2] + (8,))

    # reshape back to original residue layout
    gt_frames = fix_shape(gt_frames)
    gt_exists = fix_shape(gt_exists)
    group_exists = fix_shape(group_exists)
    residx_rigidgroup_is_ambiguous = fix_shape(residx_rigidgroup_is_ambiguous)
    alt_gt_frames = fix_shape(alt_gt_frames)

    return {
        'rigidgroups_gt_frames': gt_frames,    # Rigid (..., 8)
        'rigidgroups_gt_exists': gt_exists,    # (..., 8)
        'rigidgroups_group_exists': group_exists,    # (..., 8)
        'rigidgroups_group_is_ambiguous': residx_rigidgroup_is_ambiguous,    # (..., 8)
        'rigidgroups_alt_gt_frames': alt_gt_frames,    # Rigid (..., 8)
    }


def torsion_angles_to_frames(
        aatype: torch.Tensor,    # (N)
        backb_to_global: geometry.Rigid3Array,    # (N)
        torsion_angles_sin_cos: torch.Tensor    # (N, 7, 2)
) -> geometry.Rigid3Array:    # (N, 8)
    """Compute rigid group frames from torsion angles."""
    # Gather the default frames for all rigid groups.
    # geometry.Rigid3Array with shape (N, 8)
    m = get_rc_tensor(rc.restype_rigid_group_default_frame, aatype)
    default_frames = geometry.Rigid3Array.from_array4x4(m)

    # Create the rotation matrices according to the given angles (each frame is
    # defined such that its rotation is around the x-axis).
    sin_angles = torsion_angles_sin_cos[..., 0]
    cos_angles = torsion_angles_sin_cos[..., 1]

    # insert zero rotation for backbone group.
    num_residues = aatype.shape[-1]
    sin_angles = torch.cat([
        torch.zeros_like(aatype).unsqueeze(),
        sin_angles,
    ], dim=-1)
    cos_angles = torch.cat([torch.ones_like(aatype).unsqueeze(), cos_angles], dim=-1)
    zeros = torch.zeros_like(sin_angles)
    ones = torch.ones_like(sin_angles)

    # all_rots are geometry.Rot3Array with shape (..., N, 8)
    all_rots = geometry.Rot3Array(ones, zeros, zeros, zeros, cos_angles, -sin_angles, zeros, sin_angles, cos_angles)

    # Apply rotations to the frames.
    all_frames = default_frames.compose_rotation(all_rots)

    # chi2, chi3, and chi4 frames do not transform to the backbone frame but to
    # the previous frame. So chain them up accordingly.

    chi1_frame_to_backb = all_frames[..., 4]
    chi2_frame_to_backb = chi1_frame_to_backb @ all_frames[..., 5]
    chi3_frame_to_backb = chi2_frame_to_backb @ all_frames[..., 6]
    chi4_frame_to_backb = chi3_frame_to_backb @ all_frames[..., 7]

    all_frames_to_backb = Rigid3Array.cat([
        all_frames[..., 0:5], chi2_frame_to_backb[..., None], chi3_frame_to_backb[..., None], chi4_frame_to_backb[...,
                                                                                                                  None]
    ],
                                          dim=-1)

    # Create the global frames.
    # shape (N, 8)
    all_frames_to_global = backb_to_global[..., None] @ all_frames_to_backb

    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
        aatype: torch.Tensor,    # (*, N)
        all_frames_to_global: geometry.Rigid3Array    # (N, 8)
) -> geometry.Vec3Array:    # (*, N, 14)
    """Put atom literature positions (atom14 encoding) in each rigid group."""
    # Pick the appropriate transform for every atom.
    residx_to_group_idx = get_rc_tensor(rc.restype_atom14_to_rigid_group, aatype)
    group_mask = torch.nn.functional.one_hot(residx_to_group_idx, num_classes=8)    # shape (*, N, 14, 8)

    # geometry.Rigid3Array with shape (N, 14)
    map_atoms_to_global = all_frames_to_global[..., None, :] * group_mask
    map_atoms_to_global = map_atoms_to_global.map_tensor_fn(partial(torch.sum, dim=-1))

    # Gather the literature atom positions for each residue.
    # geometry.Vec3Array with shape (N, 14)
    lit_positions = geometry.Vec3Array.from_array(get_rc_tensor(rc.restype_atom14_rigid_group_positions, aatype))

    # Transform each atom from its local frame to the global frame.
    # geometry.Vec3Array with shape (N, 14)
    pred_positions = map_atoms_to_global.apply_to_point(lit_positions)

    # Mask out non-existing atoms.
    mask = get_rc_tensor(rc.restype_atom14_mask, aatype)
    pred_positions = pred_positions * mask

    return pred_positions


def extreme_ca_ca_distance_violations(
        positions: geometry.Vec3Array,    # (N, 37(14))
        mask: torch.Tensor,    # (N, 37(14))
        residue_index: torch.Tensor,    # (N)
        max_angstrom_tolerance=1.5,
        eps: float = 1e-6) -> torch.Tensor:
    """Counts residues whose Ca is a large distance from its neighbor."""
    this_ca_pos = positions[..., :-1, 1]    # (N - 1,)
    this_ca_mask = mask[..., :-1, 1]    # (N - 1)
    next_ca_pos = positions[..., 1:, 1]    # (N - 1,)
    next_ca_mask = mask[..., 1:, 1]    # (N - 1)
    has_no_gap_mask = ((residue_index[..., 1:] - residue_index[..., :-1]) == 1.0).astype(torch.float32)
    ca_ca_distance = geometry.euclidean_distance(this_ca_pos, next_ca_pos, eps)
    violations = (ca_ca_distance - rc.ca_ca) > max_angstrom_tolerance
    mask = this_ca_mask * next_ca_mask * has_no_gap_mask
    return tensor_utils.masked_mean(mask=mask, value=violations, dim=-1)


def get_chi_atom_indices(device: torch.device):
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
        A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
        in the order specified in rc.restypes + unknown residue type
        at the end. For chi angles which are not defined on the residue, the
        positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in rc.restypes:
        residue_name = rc.restype_1to3[residue_name]
        residue_chi_angles = rc.chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append([rc.atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append([0, 0, 0, 0])    # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)    # For UNKNOWN residue.
    return torch.tensor(chi_atom_indices, device=device)


def compute_chi_angles(positions: geometry.Vec3Array, mask: torch.Tensor, aatype: torch.Tensor):
    """Computes the chi angles given all atom positions and the amino acid type.

    Args:
        positions: A Vec3Array of shape
            [num_res, rc.atom_type_num], with positions of
            atoms needed to calculate chi angles. Supports up to 1 batch dimension.
        mask: An optional tensor of shape
            [num_res, rc.atom_type_num] that masks which atom
            positions are set for each residue. If given, then the chi mask will be
            set to 1 for a chi angle only if the amino acid has that chi angle and all
            the chi atoms needed to calculate that chi angle are set. If not given
            (set to None), the chi mask will be set to 1 for a chi angle if the amino
            acid has that chi angle and whether the actual atoms needed to calculate
            it were set will be ignored.
        aatype: A tensor of shape [num_res] with amino acid type integer
            code (0 to 21). Supports up to 1 batch dimension.

    Returns:
        A tuple of tensors (chi_angles, mask), where both have shape
        [num_res, 4]. The mask masks out unused chi angles for amino acid
        types that have less than 4 chi angles. If atom_positions_mask is set, the
        chi mask will also mask out uncomputable chi angles.
    """

    # Don't assert on the num_res and batch dimensions as they might be unknown.
    assert positions.shape[-1] == rc.atom_type_num
    assert mask.shape[-1] == rc.atom_type_num
    no_batch_dims = len(aatype.shape) - 1

    # Compute the table of chi angle indices. Shape: [restypes, chis=4, atoms=4].
    chi_atom_indices = get_chi_atom_indices(aatype.device)

    # DISCREPANCY: DeepMind doesn't remove the gaps here. I don't know why
    # theirs works.
    aatype_gapless = torch.clamp(aatype, max=20)

    # Select atoms to compute chis. Shape: [*, num_res, chis=4, atoms=4].
    atom_indices = chi_atom_indices[aatype_gapless]
    # Gather atom positions. Shape: [num_res, chis=4, atoms=4, xyz=3].
    chi_angle_atoms = positions.map_tensor_fn(
        partial(tensor_utils.batched_gather, inds=atom_indices, dim=-1, no_batch_dims=no_batch_dims + 1))

    a, b, c, d = [chi_angle_atoms[..., i] for i in range(4)]

    chi_angles = geometry.dihedral_angle(a, b, c, d)

    # Copy the chi angle mask, add the UNKNOWN residue. Shape: [restypes, 4].
    chi_angles_mask = list(rc.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = torch.tensor(chi_angles_mask, device=aatype.device)
    # Compute the chi angle mask. Shape [num_res, chis=4].
    chi_mask = chi_angles_mask[aatype_gapless]

    # The chi_mask is set to 1 only when all necessary chi angle atoms were set.
    # Gather the chi angle atoms mask. Shape: [num_res, chis=4, atoms=4].
    chi_angle_atoms_mask = tensor_utils.batched_gather(mask, atom_indices, dim=-1, no_batch_dims=no_batch_dims + 1)
    # Check if all 4 chi angle atoms were set. Shape: [num_res, chis=4].
    chi_angle_atoms_mask = torch.prod(chi_angle_atoms_mask, dim=-1)
    chi_mask = chi_mask * chi_angle_atoms_mask.to(torch.float32)

    return chi_angles, chi_mask


def make_transform_from_reference(a_xyz: geometry.Vec3Array, b_xyz: geometry.Vec3Array,
                                  c_xyz: geometry.Vec3Array) -> geometry.Rigid3Array:
    """Returns rotation and translation matrices to convert from reference.

    Note that this method does not take care of symmetries. If you provide the
    coordinates in the non-standard way, the A atom will end up in the negative
    y-axis rather than in the positive y-axis. You need to take care of such
    cases in your code.

    Args:
        a_xyz: A Vec3Array.
        b_xyz: A Vec3Array.
        c_xyz: A Vec3Array.

    Returns:
        A Rigid3Array which, when applied to coordinates in a canonicalized
        reference frame, will give coordinates approximately equal
        the original coordinates (in the global frame).
    """
    rotation = geometry.Rot3Array.from_two_vectors(c_xyz - b_xyz, a_xyz - b_xyz)
    return geometry.Rigid3Array(rotation, b_xyz)


def make_backbone_affine(
    positions: geometry.Vec3Array,
    mask: torch.Tensor,
    aatype: torch.Tensor,
) -> Tuple[geometry.Rigid3Array, torch.Tensor]:
    a = rc.atom_order['N']
    b = rc.atom_order['CA']
    c = rc.atom_order['C']

    rigid_mask = (mask[..., a] * mask[..., b] * mask[..., c])

    rigid = make_transform_from_reference(
        a_xyz=positions[..., a],
        b_xyz=positions[..., b],
        c_xyz=positions[..., c],
    )

    return rigid, rigid_mask

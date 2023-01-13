# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from fastfold.common.residue_constants import (
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
    restype_atom14_to_rigid_group,
    restype_rigid_group_default_frame,
)
from fastfold.utils.feats import frames_and_literature_positions_to_atom14_pos, torsion_angles_to_frames
from fastfold.utils.geometry.quat_rigid import QuatRigid
from fastfold.utils.geometry.rigid_matrix_vector import Rigid3Array
from fastfold.utils.geometry.vector import Vec3Array
from fastfold.utils.rigid_utils import Rigid, Rotation
from fastfold.utils.tensor_utils import dict_multimap, flatten_final_dims, permute_final_dims

from .primitives import LayerNorm, Linear, ipa_point_weights_init_


class AngleResnetBlock(nn.Module):

    def __init__(self, c_hidden):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super(AngleResnetBlock, self).__init__()

        self.c_hidden = c_hidden

        self.linear_1 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="final")

        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:

        s_initial = a

        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)

        return a + s_initial


class AngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, c_in: int, c_hidden: int, no_blocks: int, no_angles: int, epsilon: float):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
        """
        super(AngleResnet, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = Linear(self.c_hidden, self.no_angles * 2)

        self.relu = nn.ReLU()

    def forward(self, s: torch.Tensor, s_initial: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(torch.clamp(
            torch.sum(s**2, dim=-1, keepdim=True),
            min=self.eps,
        ))
        s = s / norm_denom

        return unnormalized_s, s


class PointProjection(nn.Module):

    def __init__(
        self,
        c_hidden: int,
        num_points: int,
        no_heads: int,
        return_local_points: bool = False,
    ):
        super().__init__()
        self.return_local_points = return_local_points
        self.no_heads = no_heads

        self.linear = Linear(c_hidden, no_heads * 3 * num_points)

    def forward(
        self,
        activations: torch.Tensor,
        rigids: Rigid3Array,
    ) -> Union[Vec3Array, Tuple[Vec3Array, Vec3Array]]:
        # TODO: Needs to run in high precision during training
        points_local = self.linear(activations)
        points_local = points_local.reshape(
            *points_local.shape[:-1],
            self.no_heads,
            -1,
        )
        points_local = torch.split(points_local, points_local.shape[-1] // 3, dim=-1)
        points_local = Vec3Array(*points_local)
        points_global = rigids[..., None, None].apply_to_point(points_local)

        if self.return_local_points:
            return points_global, points_local

        return points_global


class InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
        is_multimer: bool = False,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(InvariantPointAttention, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps
        self.is_multimer = is_multimer

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        if not self.is_multimer:
            hc = self.c_hidden * self.no_heads
            self.linear_q = Linear(self.c_s, hc, bias=(not is_multimer))
            self.linear_kv = Linear(self.c_s, 2 * hc)

            hpq = self.no_heads * self.no_qk_points * 3
            self.linear_q_points = Linear(self.c_s, hpq)

            hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
            self.linear_kv_points = Linear(self.c_s, hpkv)

            # hpv = self.no_heads * self.no_v_points * 3

        else:
            hc = self.c_hidden * self.no_heads
            self.linear_q = Linear(self.c_s, hc, bias=(not is_multimer))
            self.linear_q_points = PointProjection(self.c_s, self.no_qk_points, self.no_heads)

            self.linear_k = Linear(self.c_s, hc, bias=False)
            self.linear_v = Linear(self.c_s, hc, bias=False)
            self.linear_k_points = PointProjection(
                self.c_s,
                self.no_qk_points,
                self.no_heads,
            )

            self.linear_v_points = PointProjection(
                self.c_s,
                self.no_v_points,
                self.no_heads,
            )
        self.linear_b = Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (self.c_z + self.c_hidden + self.no_v_points * 4)
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        r: Union[Rigid, Rigid3Array],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        #######################################
        # Generate scalar and point activations
        #######################################

        # The following two blocks are equivalent
        # They're separated only to preserve compatibility with old AF weights
        if self.is_multimer:
            # [*, N_res, H * C_hidden]
            q = self.linear_q(s)

            # [*, N_res, H, C_hidden]
            q = q.view(q.shape[:-1] + (self.no_heads, -1))

            # [*, N_res, H, P_qk]
            q_pts = self.linear_q_points(s, r)
            # [*, N_res, H * C_hidden]
            k = self.linear_k(s)
            v = self.linear_v(s)

            # [*, N_res, H, C_hidden]
            k = k.view(k.shape[:-1] + (self.no_heads, -1))
            v = v.view(v.shape[:-1] + (self.no_heads, -1))

            # [*, N_res, H, P_qk, 3]
            k_pts = self.linear_k_points(s, r)

            # [*, N_res, H, P_v, 3]
            v_pts = self.linear_v_points(s, r)
        else:
            # [*, N_res, H * C_hidden]
            q = self.linear_q(s)
            kv = self.linear_kv(s)

            # [*, N_res, H, C_hidden]
            q = q.view(q.shape[:-1] + (self.no_heads, -1))

            # [*, N_res, H, 2 * C_hidden]
            kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

            # [*, N_res, H, C_hidden]
            k, v = torch.split(kv, self.c_hidden, dim=-1)

            # [*, N_res, H * P_q * 3]
            q_pts = self.linear_q_points(s)

            # This is kind of clunky, but it's how the original does it
            # [*, N_res, H * P_q, 3]
            q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
            q_pts = torch.stack(q_pts, dim=-1)
            q_pts = r[..., None].apply(q_pts)

            # [*, N_res, H, P_q, 3]
            q_pts = q_pts.view(q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))

            # [*, N_res, H * (P_q + P_v) * 3]
            kv_pts = self.linear_kv_points(s)

            # [*, N_res, H * (P_q + P_v), 3]
            kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
            kv_pts = torch.stack(kv_pts, dim=-1)
            kv_pts = r[..., None].apply(kv_pts)

            # [*, N_res, H, (P_q + P_v), 3]
            kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

            # [*, N_res, H, P_q/P_v, 3]
            k_pts, v_pts = torch.split(kv_pts, [self.no_qk_points, self.no_v_points], dim=-2)

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z)

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),    # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),    # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        if self.is_multimer:
            # [*, N_res, N_res, H, P_q, 3]
            pt_att = q_pts[..., None, :, :] - k_pts[..., None, :, :, :]
            # [*, N_res, N_res, H, P_q]
            pt_att = sum([c**2 for c in pt_att])
        else:
            # [*, N_res, N_res, H, P_q, 3]
            pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
            pt_att = pt_att**2
            # [*, N_res, N_res, H, P_q]
            pt_att = sum(torch.unbind(pt_att, dim=-1))

        head_weights = self.softplus(self.head_weights).view(*((1,) * len(pt_att.shape[:-2]) + (-1, 1)))
        head_weights = head_weights * math.sqrt(1.0 / (3 * (self.no_qk_points * 9.0 / 2)))
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # As DeepMind explains, this manual matmul ensures that the operation
        # happens in float32.
        if self.is_multimer:
            # [*, N_res, H, P_v]
            o_pt = v_pts * permute_final_dims(a, (1, 2, 0)).unsqueeze(-1)
            o_pt = o_pt.sum(dim=-3)

            # [*, N_res, H, P_v]
            o_pt = r[..., None, None].apply_inverse_to_point(o_pt)

            # [*, N_res, H * P_v, 3]
            o_pt = o_pt.reshape(o_pt.shape[:-2] + (-1,))

            # [*, N_res, H * P_v]
            o_pt_norm = o_pt.norm(self.eps)
        else:
            # [*, H, 3, N_res, P_v]
            o_pt = torch.sum(
                (a[..., None, :, :, None] * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]),
                dim=-2,
            )

            # [*, N_res, H, P_v, 3]
            o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
            o_pt = r[..., None, None].invert_apply(o_pt)

            # [*, N_res, H * P_v]
            o_pt_norm = flatten_final_dims(torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps), 2)

            # [*, N_res, H * P_v, 3]
            o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # [*, N_res, H, C_z]
        o_pair = torch.matmul(a.transpose(-2, -3), z.to(dtype=a.dtype))

        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        # [*, N_res, C_s]
        if self.is_multimer:
            s = self.linear_out(torch.cat((o, *o_pt, o_pt_norm, o_pair), dim=-1).to(dtype=z.dtype))
        else:
            s = self.linear_out(
                torch.cat((o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1).to(dtype=z.dtype))

        return s


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s: int):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s

        self.linear = Linear(self.c_s, 6, init="final")

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        # [*, 6]
        update = self.linear(s)

        return update


class StructureModuleTransitionLayer(nn.Module):

    def __init__(self, c: int):
        super(StructureModuleTransitionLayer, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")

        self.relu = nn.ReLU()

    def forward(self, s: torch.Tensor):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s


class StructureModuleTransition(nn.Module):

    def __init__(self, c: int, num_layers: int, dropout_rate: float):
        super(StructureModuleTransition, self).__init__()

        self.c = c
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            l = StructureModuleTransitionLayer(self.c)
            self.layers.append(l)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = LayerNorm(self.c)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            s = l(s)

        s = self.dropout(s)
        s = self.layer_norm(s)

        return s


class StructureModule(nn.Module):

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_ipa: int,
        c_resnet: int,
        no_heads_ipa: int,
        no_qk_points: int,
        no_v_points: int,
        dropout_rate: float,
        no_blocks: int,
        no_transition_layers: int,
        no_resnet_blocks: int,
        no_angles: int,
        trans_scale_factor: float,
        epsilon: float,
        inf: float,
        is_multimer: bool = False,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_ipa:
                IPA hidden channel dimension
            c_resnet:
                Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
            no_heads_ipa:
                Number of IPA heads
            no_qk_points:
                Number of query/key points to generate during IPA
            no_v_points:
                Number of value points to generate during IPA
            dropout_rate:
                Dropout rate used throughout the layer
            no_blocks:
                Number of structure module blocks
            no_transition_layers:
                Number of layers in the single representation transition
                (Alg. 23 lines 8-9)
            no_resnet_blocks:
                Number of blocks in the angle resnet
            no_angles:
                Number of angles to generate in the angle resnet
            trans_scale_factor:
                Scale of single representation transition hidden dimension
            epsilon:
                Small number used in angle resnet normalization
            inf:
                Large number used for attention masking
            is_multimer:
                whether running under multimer mode
        """
        super(StructureModule, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_ipa = c_ipa
        self.c_resnet = c_resnet
        self.no_heads_ipa = no_heads_ipa
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.dropout_rate = dropout_rate
        self.no_blocks = no_blocks
        self.no_transition_layers = no_transition_layers
        self.no_resnet_blocks = no_resnet_blocks
        self.no_angles = no_angles
        self.trans_scale_factor = trans_scale_factor
        self.epsilon = epsilon
        self.inf = inf
        self.is_multimer = is_multimer

        # To be lazily initialized later
        self.default_frames = None
        self.group_idx = None
        self.atom_mask = None
        self.lit_positions = None

        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)

        self.linear_in = Linear(self.c_s, self.c_s)

        self.ipa = InvariantPointAttention(
            self.c_s,
            self.c_z,
            self.c_ipa,
            self.no_heads_ipa,
            self.no_qk_points,
            self.no_v_points,
            inf=self.inf,
            eps=self.epsilon,
            is_multimer=self.is_multimer,
        )

        self.ipa_dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm_ipa = LayerNorm(self.c_s)

        self.transition = StructureModuleTransition(
            self.c_s,
            self.no_transition_layers,
            self.dropout_rate,
        )

        if is_multimer:
            self.bb_update = QuatRigid(self.c_s, full_quat=False)
        else:
            self.bb_update = BackboneUpdate(self.c_s)

        self.angle_resnet = AngleResnet(
            self.c_s,
            self.c_resnet,
            self.no_resnet_blocks,
            self.no_angles,
            self.epsilon,
        )

    def _forward_monomer(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        aatype: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        if mask is None:
            # [*, N]
            mask = s.new_ones(s.shape[:-1])

        # [*, N, C_s]
        s = self.layer_norm_s(s)

        # [*, N, N, C_z]
        z = self.layer_norm_z(z)

        # [*, N, C_s]
        s_initial = s
        s = self.linear_in(s)

        # [*, N]
        rigids = Rigid.identity(
            s.shape[:-1],
            s.dtype,
            s.device,
            self.training,
            fmt="quat",
        )
        outputs = []
        for i in range(self.no_blocks):
            # [*, N, C_s]
            s = s + self.ipa(s, z, rigids, mask)
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            # [*, N]
            rigids = rigids.compose_q_update_vec(self.bb_update(s))

            # To hew as closely as possible to AlphaFold, we convert our
            # quaternion-based transformations to rotation-matrix ones
            # here
            backb_to_global = Rigid(
                Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
                rigids.get_trans(),
            )

            backb_to_global = backb_to_global.scale_translation(self.trans_scale_factor)

            # [*, N, 7, 2]
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)

            all_frames_to_global = self.torsion_angles_to_frames(
                backb_to_global,
                angles,
                aatype,
            )

            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                aatype,
            )

            scaled_rigids = rigids.scale_translation(self.trans_scale_factor)

            preds = {
                "frames": scaled_rigids.to_tensor_7(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
            }

            outputs.append(preds)

            if i < (self.no_blocks - 1):
                rigids = rigids.stop_rot_gradient()

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s

        return outputs

    def _forward_multimer(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        aatype: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if mask is None:
            # [*, N]
            mask = s.new_ones(s.shape[:-1])

        # [*, N, C_s]
        s = self.layer_norm_s(s)

        # [*, N, N, C_z]
        z = self.layer_norm_z(z)

        # [*, N, C_s]
        s_initial = s
        s = self.linear_in(s)

        # [*, N]
        rigids = Rigid3Array.identity(
            s.shape[:-1],
            s.device,
        )
        outputs = []
        for i in range(self.no_blocks):
            # [*, N, C_s]
            s = s + self.ipa(s, z, rigids, mask)
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            # [*, N]
            rigids = rigids @ self.bb_update(s)

            # [*, N, 7, 2]
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)

            all_frames_to_global = self.torsion_angles_to_frames(
                rigids.scale_translation(self.trans_scale_factor),
                angles,
                aatype,
            )

            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                aatype,
            )

            preds = {
                "frames": rigids.scale_translation(self.trans_scale_factor).to_tensor(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz.to_tensor(),
            }

            outputs.append(preds)

            if i < (self.no_blocks - 1):
                rigids = rigids.stop_rot_gradient()

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s

        return outputs

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        aatype: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        if self.is_multimer:
            outputs = self._forward_multimer(s, z, aatype, mask)
        else:
            outputs = self._forward_monomer(s, z, aatype, mask)

        return outputs

    def _init_residue_constants(self, float_dtype: torch.dtype, device: torch.device):
        if self.default_frames is None:
            self.default_frames = torch.tensor(
                restype_rigid_group_default_frame,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.group_idx is None:
            self.group_idx = torch.tensor(
                restype_atom14_to_rigid_group,
                device=device,
                requires_grad=False,
            )
        if self.atom_mask is None:
            self.atom_mask = torch.tensor(
                restype_atom14_mask,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.lit_positions is None:
            self.lit_positions = torch.tensor(
                restype_atom14_rigid_group_positions,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )

    def torsion_angles_to_frames(self, r: Union[Rigid, Rigid3Array], alpha: torch.Tensor, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
            self,
            r: Union[Rigid, Rigid3Array],
            f    # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        if type(r) == Rigid:
            self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        elif type(r) == Rigid3Array:
            self._init_residue_constants(r.dtype, r.device)
        else:
            raise ValueError("Unknown rigid type")
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )

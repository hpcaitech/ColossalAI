from functools import partial
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .primitives import LayerNorm, Linear
from .template import TemplatePairStack, TemplatePointwiseAttention
from .utils import all_atom_multimer, dgram_from_positions, geometry
from .utils.tensor_utils import dict_multimap, one_hot, tensor_tree_map


class InputEmbedderMultimer(nn.Module):
    """
    Embeds a subset of the input features.

    Implements Algorithms 3 (InputEmbedder) and 4 (relpos).
    """

    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        c_z: int,
        c_m: int,
        max_relative_idx: int,
        use_chain_relative: bool,
        max_relative_chain: int,
        **kwargs,
    ):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            msa_dim:
                Final dimension of the MSA features
            c_z:
                Pair embedding dimension
            c_m:
                MSA embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super(InputEmbedderMultimer, self).__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim

        self.c_z = c_z
        self.c_m = c_m

        self.linear_tf_z_i = Linear(tf_dim, c_z)
        self.linear_tf_z_j = Linear(tf_dim, c_z)
        self.linear_tf_m = Linear(tf_dim, c_m)
        self.linear_msa_m = Linear(msa_dim, c_m)

        # RPE stuff
        self.max_relative_idx = max_relative_idx
        self.use_chain_relative = use_chain_relative
        self.max_relative_chain = max_relative_chain
        if self.use_chain_relative:
            self.no_bins = 2 * max_relative_idx + 2 + 1 + 2 * max_relative_chain + 2
        else:
            self.no_bins = 2 * max_relative_idx + 1
        self.linear_relpos = Linear(self.no_bins, c_z)

    def relpos(self, batch: Dict[str, torch.Tensor]):
        pos = batch["residue_index"]
        asym_id = batch["asym_id"]
        asym_id_same = asym_id[..., None] == asym_id[..., None, :]
        offset = pos[..., None] - pos[..., None, :]

        clipped_offset = torch.clamp(offset + self.max_relative_idx, 0, 2 * self.max_relative_idx)

        rel_feats = []
        if self.use_chain_relative:
            final_offset = torch.where(
                asym_id_same,
                clipped_offset,
                (2 * self.max_relative_idx + 1) * torch.ones_like(clipped_offset),
            )

            rel_pos = torch.nn.functional.one_hot(
                final_offset,
                2 * self.max_relative_idx + 2,
            )

            rel_feats.append(rel_pos)

            entity_id = batch["entity_id"]
            entity_id_same = entity_id[..., None] == entity_id[..., None, :]
            rel_feats.append(entity_id_same[..., None])

            sym_id = batch["sym_id"]
            rel_sym_id = sym_id[..., None] - sym_id[..., None, :]

            max_rel_chain = self.max_relative_chain
            clipped_rel_chain = torch.clamp(
                rel_sym_id + max_rel_chain,
                0,
                2 * max_rel_chain,
            )

            final_rel_chain = torch.where(
                entity_id_same,
                clipped_rel_chain,
                (2 * max_rel_chain + 1) * torch.ones_like(clipped_rel_chain),
            )

            rel_chain = torch.nn.functional.one_hot(
                final_rel_chain.long(),
                2 * max_rel_chain + 2,
            )

            rel_feats.append(rel_chain)
        else:
            rel_pos = torch.nn.functional.one_hot(
                clipped_offset,
                2 * self.max_relative_idx + 1,
            )
            rel_feats.append(rel_pos)

        rel_feat = torch.cat(rel_feats, dim=-1).to(self.linear_relpos.weight.dtype)

        return self.linear_relpos(rel_feat)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        tf = batch["target_feat"]
        msa = batch["msa_feat"]

        # [*, N_res, c_z]
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)

        # [*, N_res, N_res, c_z]
        pair_emb = tf_emb_i[..., None, :] + tf_emb_j[..., None, :, :]
        pair_emb = pair_emb + self.relpos(batch)

        # [*, N_clust, N_res, c_m]
        n_clust = msa.shape[-3]
        tf_m = (self.linear_tf_m(tf).unsqueeze(-3).expand(((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1))))
        msa_emb = self.linear_msa_m(msa) + tf_m

        return msa_emb, pair_emb


class TemplatePairEmbedderMultimer(nn.Module):

    def __init__(
        self,
        c_z: int,
        c_out: int,
        c_dgram: int,
        c_aatype: int,
    ):
        super().__init__()

        self.dgram_linear = Linear(c_dgram, c_out)
        self.aatype_linear_1 = Linear(c_aatype, c_out)
        self.aatype_linear_2 = Linear(c_aatype, c_out)
        self.query_embedding_layer_norm = LayerNorm(c_z)
        self.query_embedding_linear = Linear(c_z, c_out)

        self.pseudo_beta_mask_linear = Linear(1, c_out)
        self.x_linear = Linear(1, c_out)
        self.y_linear = Linear(1, c_out)
        self.z_linear = Linear(1, c_out)
        self.backbone_mask_linear = Linear(1, c_out)

    def forward(
        self,
        template_dgram: torch.Tensor,
        aatype_one_hot: torch.Tensor,
        query_embedding: torch.Tensor,
        pseudo_beta_mask: torch.Tensor,
        backbone_mask: torch.Tensor,
        multichain_mask_2d: torch.Tensor,
        unit_vector: geometry.Vec3Array,
    ) -> torch.Tensor:
        act = 0.

        pseudo_beta_mask_2d = (pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :])
        pseudo_beta_mask_2d *= multichain_mask_2d
        template_dgram *= pseudo_beta_mask_2d[..., None]
        act += self.dgram_linear(template_dgram)
        act += self.pseudo_beta_mask_linear(pseudo_beta_mask_2d[..., None])

        aatype_one_hot = aatype_one_hot.to(template_dgram.dtype)
        act += self.aatype_linear_1(aatype_one_hot[..., None, :, :])
        act += self.aatype_linear_2(aatype_one_hot[..., None, :])

        backbone_mask_2d = (backbone_mask[..., None] * backbone_mask[..., None, :])
        backbone_mask_2d *= multichain_mask_2d
        x, y, z = [coord * backbone_mask_2d for coord in unit_vector]
        act += self.x_linear(x[..., None])
        act += self.y_linear(y[..., None])
        act += self.z_linear(z[..., None])

        act += self.backbone_mask_linear(backbone_mask_2d[..., None])

        query_embedding = self.query_embedding_layer_norm(query_embedding)
        act += self.query_embedding_linear(query_embedding)

        return act


class TemplateSingleEmbedderMultimer(nn.Module):

    def __init__(
        self,
        c_in: int,
        c_m: int,
    ):
        super().__init__()
        self.template_single_embedder = Linear(c_in, c_m)
        self.template_projector = Linear(c_m, c_m)

    def forward(
        self,
        batch,
        atom_pos,
        aatype_one_hot,
    ):
        out = {}

        template_chi_angles, template_chi_mask = (all_atom_multimer.compute_chi_angles(
            atom_pos,
            batch["template_all_atom_mask"],
            batch["template_aatype"],
        ))

        template_features = torch.cat(
            [
                aatype_one_hot,
                torch.sin(template_chi_angles) * template_chi_mask,
                torch.cos(template_chi_angles) * template_chi_mask,
                template_chi_mask,
            ],
            dim=-1,
        )

        template_mask = template_chi_mask[..., 0]

        template_activations = self.template_single_embedder(template_features)
        template_activations = torch.nn.functional.relu(template_activations)
        template_activations = self.template_projector(template_activations,)

        out["template_single_embedding"] = (template_activations)
        out["template_mask"] = template_mask

        return out


class TemplateEmbedderMultimer(nn.Module):

    def __init__(self, config):
        super(TemplateEmbedderMultimer, self).__init__()

        self.config = config
        self.template_pair_embedder = TemplatePairEmbedderMultimer(**config["template_pair_embedder"],)
        self.template_single_embedder = TemplateSingleEmbedderMultimer(**config["template_single_embedder"],)
        self.template_pair_stack = TemplatePairStack(**config["template_pair_stack"],)

        self.linear_t = Linear(config.c_t, config.c_z)

    def forward(
        self,
        batch,
        z,
        padding_mask_2d,
        templ_dim,
        chunk_size,
        multichain_mask_2d,
    ):
        template_embeds = []
        n_templ = batch["template_aatype"].shape[templ_dim]
        for i in range(n_templ):
            idx = batch["template_aatype"].new_tensor(i)
            single_template_feats = tensor_tree_map(
                lambda t: torch.index_select(t, templ_dim, idx),
                batch,
            )

            single_template_embeds = {}
            act = 0.

            template_positions, pseudo_beta_mask = (
                single_template_feats["template_pseudo_beta"],
                single_template_feats["template_pseudo_beta_mask"],
            )

            template_dgram = dgram_from_positions(
                template_positions,
                inf=self.config.inf,
                **self.config.distogram,
            )

            aatype_one_hot = torch.nn.functional.one_hot(
                single_template_feats["template_aatype"],
                22,
            )

            raw_atom_pos = single_template_feats["template_all_atom_positions"]

            atom_pos = geometry.Vec3Array.from_array(raw_atom_pos)
            rigid, backbone_mask = all_atom_multimer.make_backbone_affine(
                atom_pos,
                single_template_feats["template_all_atom_mask"],
                single_template_feats["template_aatype"],
            )
            points = rigid.translation
            rigid_vec = rigid[..., None].inverse().apply_to_point(points)
            unit_vector = rigid_vec.normalized()

            pair_act = self.template_pair_embedder(
                template_dgram,
                aatype_one_hot,
                z,
                pseudo_beta_mask,
                backbone_mask,
                multichain_mask_2d,
                unit_vector,
            )

            single_template_embeds["template_pair_embedding"] = pair_act
            single_template_embeds.update(
                self.template_single_embedder(
                    single_template_feats,
                    atom_pos,
                    aatype_one_hot,
                ))
            template_embeds.append(single_template_embeds)

        template_embeds = dict_multimap(
            partial(torch.cat, dim=templ_dim),
            template_embeds,
        )

        # [*, S_t, N, N, C_z]
        t = self.template_pair_stack(
            template_embeds["template_pair_embedding"],
            padding_mask_2d.unsqueeze(-3).to(dtype=z.dtype),
            chunk_size=chunk_size,
            _mask_trans=False,
        )
        # [*, N, N, C_z]
        t = torch.sum(t, dim=-4) / n_templ
        t = torch.nn.functional.relu(t)
        t = self.linear_t(t)
        template_embeds["template_pair_embedding"] = t

        return template_embeds

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
import torch
import torch.nn as nn
from typing import Tuple, Optional
from functools import partial

from openfold.primitives import Linear, LayerNorm
from openfold.dropout import DropoutRowwise, DropoutColumnwise
from openfold.msa import (
    MSARowAttentionWithPairBias,
    MSAColumnAttention,
    MSAColumnGlobalAttention,
)
from openfold.outer_product_mean import OuterProductMean
from openfold.pair_transition import PairTransition
from openfold.triangular_attention import (
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
)
from openfold.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from openfold.checkpointing import checkpoint_blocks, get_checkpoint_fn
from openfold.tensor_utils import chunk_layer


class MSATransition(nn.Module):
    """
    Feed-forward network applied to MSA activations after attention.

    Implements Algorithm 9
    """
    def __init__(self, c_m, n):
        """
        Args:
            c_m:
                MSA channel dimension
            n:
                Factor multiplied to c_m to obtain the hidden channel
                dimension
        """
        super(MSATransition, self).__init__()

        self.c_m = c_m
        self.n = n

        self.layer_norm = LayerNorm(self.c_m)
        self.linear_1 = Linear(self.c_m, self.n * self.c_m, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_m, self.c_m, init="final")

    def _transition(self, m, mask):
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m) * mask
        return m

    @torch.jit.ignore
    def _chunk(self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
         return chunk_layer(
             self._transition,
             {"m": m, "mask": mask},
             chunk_size=chunk_size,
             no_batch_dims=len(m.shape[:-2]),
         )

    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA activation
            mask:
                [*, N_seq, N_res, C_m] MSA mask
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA activation update
        """

        # DISCREPANCY: DeepMind forgets to apply the MSA mask here.
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        # [*, N_seq, N_res, 1]
        mask = mask.unsqueeze(-1)

        m = self.layer_norm(m)

        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size)
        else:
            m = self._transition(m, mask)

        return m


class EvoformerBlockCore(nn.Module):
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float,
        inf: float,
        eps: float,
        _is_extra_msa_stack: bool = False,
        is_multimer: bool = False,
    ):
        super(EvoformerBlockCore, self).__init__()
        self.is_multimer = is_multimer
        self.msa_transition = MSATransition(
            c_m=c_m,
            n=transition_n,
        )

        self.outer_product_mean = OuterProductMean(
            c_m,
            c_z,
            c_hidden_opm,
        )

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z,
            c_hidden_mul,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_z,
            c_hidden_mul,
        )

        self.tri_att_start = TriangleAttentionStartingNode(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )

        self.pair_transition = PairTransition(
            c_z,
            transition_n,
        )

        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)
        self.ps_dropout_col_layer = DropoutColumnwise(pair_dropout)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]: 
        # DeepMind doesn't mask these transitions in the source, so _mask_trans
        # should be disabled to better approximate the exact activations of
        # the original.
        msa_trans_mask = msa_mask if _mask_trans else None
        pair_trans_mask = pair_mask if _mask_trans else None

        m = m + self.msa_transition(
            m, mask=msa_trans_mask, chunk_size=chunk_size
        )
        z = z + self.outer_product_mean(
            m, mask=msa_mask, chunk_size=chunk_size
        )
        z = z + self.ps_dropout_row_layer(self.tri_mul_out(z, mask=pair_mask))
        z = z + self.ps_dropout_row_layer(self.tri_mul_in(z, mask=pair_mask))
        z = z + self.ps_dropout_row_layer(
            self.tri_att_start(z, mask=pair_mask, chunk_size=chunk_size)
        )
        z = z + self.ps_dropout_col_layer(
            self.tri_att_end(z, mask=pair_mask, chunk_size=chunk_size)
        )
        z = z + self.pair_transition(
            z, mask=pair_trans_mask, chunk_size=chunk_size
        )

        return m, z


class EvoformerBlock(nn.Module):
    def __init__(self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        inf: float,
        eps: float,
        is_multimer: bool,
    ):
        super(EvoformerBlock, self).__init__()

        self.msa_att_row = MSARowAttentionWithPairBias(
            c_m=c_m,
            c_z=c_z,
            c_hidden=c_hidden_msa_att,
            no_heads=no_heads_msa,
            inf=inf,
        )

        self.msa_att_col = MSAColumnAttention(
            c_m,
            c_hidden_msa_att,
            no_heads_msa,
            inf=inf,
        )

        self.msa_dropout_layer = DropoutRowwise(msa_dropout)

        self.core = EvoformerBlockCore(
            c_m=c_m,
            c_z=c_z,
            c_hidden_opm=c_hidden_opm,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_msa=no_heads_msa,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
            eps=eps,
        )
        
        self.outer_product_mean = OuterProductMean(
            c_m,
            c_z,
            c_hidden_opm,
        )
        self.is_multimer = is_multimer

    def forward(self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m = m + self.msa_dropout_layer(
            self.msa_att_row(m, z=z, mask=msa_mask, chunk_size=chunk_size)
        )
        m = m + self.msa_att_col(m, mask=msa_mask, chunk_size=chunk_size)
        m, z = self.core(
            m, 
            z, 
            msa_mask=msa_mask, 
            pair_mask=pair_mask, 
            chunk_size=chunk_size, 
            _mask_trans=_mask_trans,
        )

        return m, z


class ExtraMSABlock(nn.Module):
    """ 
        Almost identical to the standard EvoformerBlock, except in that the
        ExtraMSABlock uses GlobalAttention for MSA column attention and
        requires more fine-grained control over checkpointing. Separated from
        its twin to preserve the TorchScript-ability of the latter.
    """
    def __init__(self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        inf: float,
        eps: float,
        ckpt: bool,
        is_multimer: bool,
    ):
        super(ExtraMSABlock, self).__init__()
        
        self.ckpt = ckpt

        self.msa_att_row = MSARowAttentionWithPairBias(
            c_m=c_m,
            c_z=c_z,
            c_hidden=c_hidden_msa_att,
            no_heads=no_heads_msa,
            inf=inf,
        )

        self.msa_att_col = MSAColumnGlobalAttention(
            c_in=c_m,
            c_hidden=c_hidden_msa_att,
            no_heads=no_heads_msa,
            inf=inf,
            eps=eps,
        )

        self.msa_dropout_layer = DropoutRowwise(msa_dropout)

        self.core = EvoformerBlockCore(
            c_m=c_m,
            c_z=c_z,
            c_hidden_opm=c_hidden_opm,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_msa=no_heads_msa,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
            eps=eps,
        )
        self.is_multimer = is_multimer

    def forward(self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        _chunk_logits: Optional[int] = 1024,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m = m + self.msa_dropout_layer(
            self.msa_att_row(
                m.clone(), 
                z=z.clone(), 
                mask=msa_mask, 
                chunk_size=chunk_size,
                _chunk_logits=_chunk_logits if torch.is_grad_enabled() else None,
                _checkpoint_chunks=
                    self.ckpt if torch.is_grad_enabled() else False,
            )
        )

        def fn(m, z):
            m = m + self.msa_att_col(m, mask=msa_mask, chunk_size=chunk_size)
            m, z = self.core(
                m, z, msa_mask=msa_mask, pair_mask=pair_mask, chunk_size=chunk_size
            )
            
            return m, z

        if(torch.is_grad_enabled() and self.ckpt):
            checkpoint_fn = get_checkpoint_fn()
            m, z = checkpoint_fn(fn, m, z)
        else:
            m, z = fn(m, z)

        return m, z


class EvoformerStack(nn.Module):
    """
    Main Evoformer trunk.

    Implements Algorithm 6.
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        c_s: int,
        no_heads_msa: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        blocks_per_ckpt: int,
        inf: float,
        eps: float,
        clear_cache_between_blocks: bool = False, 
        is_multimer: bool = False,
        **kwargs,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair channel dimension
            c_hidden_msa_att:
                Hidden dimension in MSA attention
            c_hidden_opm:
                Hidden dimension in outer product mean module
            c_hidden_mul:
                Hidden dimension in multiplicative updates
            c_hidden_pair_att:
                Hidden dimension in triangular attention
            c_s:
                Channel dimension of the output "single" embedding
            no_heads_msa:
                Number of heads used for MSA attention
            no_heads_pair:
                Number of heads used for pair attention
            no_blocks:
                Number of Evoformer blocks in the stack
            transition_n:
                Factor by which to multiply c_m to obtain the MSATransition
                hidden dimension
            msa_dropout:
                Dropout rate for MSA activations
            pair_dropout:
                Dropout used for pair activations
            blocks_per_ckpt:
                Number of Evoformer blocks in each activation checkpoint
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation
        """
        super(EvoformerStack, self).__init__()

        self.blocks_per_ckpt = blocks_per_ckpt
        self.clear_cache_between_blocks = clear_cache_between_blocks

        self.blocks = nn.ModuleList()

        for _ in range(no_blocks):
            block = EvoformerBlock(
                c_m=c_m,
                c_z=c_z,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_msa=no_heads_msa,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                inf=inf,
                eps=eps,
                is_multimer=is_multimer,
            )
            self.blocks.append(block)

        self.linear = Linear(c_m, c_s)

    def forward(self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            msa_mask:
                [*, N_seq, N_res] MSA mask
            pair_mask:
                [*, N_res, N_res] pair mask
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            s:
                [*, N_res, C_s] single embedding (or None if extra MSA stack)
        """
        blocks = [
            partial(
                b,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        if(self.clear_cache_between_blocks):
            def block_with_cache_clear(block, *args):
                torch.cuda.empty_cache()
                return block(*args)

            blocks = [partial(block_with_cache_clear, b) for b in blocks]

        m, z = checkpoint_blocks(
            blocks,
            args=(m, z),
            blocks_per_ckpt=self.blocks_per_ckpt if self.training else None,
        )

        s = self.linear(m[..., 0, :, :])
        
        return m, z, s


class ExtraMSAStack(nn.Module):
    """
    Implements Algorithm 18.
    """

    def __init__(self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        inf: float,
        eps: float,
        ckpt: bool,
        clear_cache_between_blocks: bool = False,
        is_multimer: bool = False,
        **kwargs,
    ):
        super(ExtraMSAStack, self).__init__()
        
        self.clear_cache_between_blocks = clear_cache_between_blocks
        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = ExtraMSABlock(
                c_m=c_m,
                c_z=c_z,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_msa=no_heads_msa,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                inf=inf,
                eps=eps,
                ckpt=ckpt,
                is_multimer=is_multimer,
            )
            self.blocks.append(block)

    def forward(self,
        m: torch.Tensor,
        z: torch.Tensor,
        chunk_size: int,
        msa_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_extra, N_res, C_m] extra MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            msa_mask:
                Optional [*, N_extra, N_res] MSA mask
            pair_mask:
                Optional [*, N_res, N_res] pair mask
        Returns:
            [*, N_res, N_res, C_z] pair update
        """ 
        #checkpoint_fn = get_checkpoint_fn()
        #blocks = [
        #    partial(b, msa_mask=msa_mask, pair_mask=pair_mask, chunk_size=chunk_size, _chunk_logits=None) for b in self.blocks
        #]

        #def dodo(b, *args):
        #    torch.cuda.empty_cache()
        #    return b(*args)

        #blocks = [partial(dodo, b) for b in blocks]

        #for b in blocks:
        #    if(torch.is_grad_enabled()):
        #        m, z = checkpoint_fn(b, *(m, z))
        #    else:
        #        m, z = b(m, z)

        for b in self.blocks:
            m, z = b(m, z, msa_mask, pair_mask, chunk_size=chunk_size)

            if(self.clear_cache_between_blocks):
                torch.cuda.empty_cache()

        return z
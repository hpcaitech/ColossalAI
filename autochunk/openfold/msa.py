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
from typing import Optional, List, Tuple

from openfold.primitives import (
    Linear, 
    LayerNorm,
    Attention, 
    GlobalAttention, 
    _attention_chunked_trainable,
)
from openfold.checkpointing import get_checkpoint_fn
from openfold.tensor_utils import (
    chunk_layer,
    permute_final_dims,
    flatten_final_dims,
)


class MSAAttention(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        no_heads,
        pair_bias=False,
        c_z=None,
        inf=1e9,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            pair_bias:
                Whether to use pair embedding bias
            c_z:
                Pair embedding channel dimension. Ignored unless pair_bias
                is true
            inf:
                A large number to be used in computing the attention mask
        """
        super(MSAAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.pair_bias = pair_bias
        self.c_z = c_z
        self.inf = inf

        self.layer_norm_m = LayerNorm(self.c_in)

        self.layer_norm_z = None
        self.linear_z = None
        if self.pair_bias:
            self.layer_norm_z = LayerNorm(self.c_z)
            self.linear_z = Linear(
                self.c_z, self.no_heads, bias=False, init="normal"
            )
        
        self.mha = Attention(
            self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads
        )

    @torch.jit.ignore
    def _chunk(self, 
        m: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
    ) -> torch.Tensor:
        return chunk_layer(
            self.mha,
            {"q_x": m, "kv_x": m, "biases": biases},
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2]),
        )

    def _prep_inputs(self,
        m: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [*, N_seq, N_res, C_m]
        m = self.layer_norm_m(m)

        n_seq, n_res = m.shape[-3:-1]
        if mask is None:
            # [*, N_seq, N_res]
            mask = m.new_ones(
                m.shape[:-3] + (n_seq, n_res),
            )

        # [*, N_seq, 1, 1, N_res]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # This step simply returns a larger view of the bias, and does not
        # consume additional memory.
        # [*, N_seq, no_heads, N_res, N_res]
        #bias = bias.expand(
        #    ((-1,) * len(bias.shape[:-4])) + (-1, self.no_heads, n_res, -1)
        #)

        if (self.pair_bias and 
            z is not None and                       # For the 
            self.layer_norm_z is not None and       # benefit of
            self.linear_z is not None               # TorchScript
        ):
            # [*, N_res, N_res, C_z]
            z = self.layer_norm_z(z)
            
            # [*, N_res, N_res, no_heads]
            z = self.linear_z(z)
            
            # [*, 1, no_heads, N_res, N_res]
            z = permute_final_dims(z, (2, 0, 1)).unsqueeze(-4)

        return m, mask_bias, z


    def forward(self, 
        m: torch.Tensor, 
        z: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None, 
        chunk_size: Optional[int] = None,
        _chunk_logits: Optional[int] = None,
        _checkpoint_chunks: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding. Required only if
                pair_bias is True
            mask:
                [*, N_seq, N_res] MSA mask
            chunk_size:
                Size of chunks into which the inputs are split along their
                batch dimensions. A low value decreases memory overhead at the 
                cost of slower execution. Chunking is not performed by default.
                
        """
        m, mask_bias, z = self._prep_inputs(m, z, mask)

        biases = [mask_bias]
        if(z is not None):
            biases.append(z)

        if chunk_size is not None:
            m = self._chunk(m, biases, chunk_size)
        else:
            m = self.mha(
                q_x=m, 
                kv_x=m, 
                biases=biases 
            )

        return m


class MSARowAttentionWithPairBias(MSAAttention):
    """
    Implements Algorithm 7.
    """

    def __init__(self, c_m, c_z, c_hidden, no_heads, inf=1e9):
        """
        Args:
            c_m:
                Input channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            inf:
                Large number used to construct attention masks
        """
        super(MSARowAttentionWithPairBias, self).__init__(
            c_m,
            c_hidden,
            no_heads,
            pair_bias=True,
            c_z=c_z,
            inf=inf,
        )


class MSAColumnAttention(nn.Module):
    """
    Implements Algorithm 8.

    By rights, this should also be a subclass of MSAAttention. Alas,
    most inheritance isn't supported by TorchScript.
    """

    def __init__(self, c_m, c_hidden, no_heads, inf=1e9):
        """
        Args:
            c_m:
                MSA channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            inf:
                Large number used to construct attention masks
        """
        super(MSAColumnAttention, self).__init__()
        
        self.c_m = c_m
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self._msa_att = MSAAttention(
            c_in=c_m,
            c_hidden=c_hidden,
            no_heads=no_heads,
            pair_bias=False,
            c_z=None,
            inf=inf,
        )

    def forward(self, 
        m: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            mask:
                [*, N_seq, N_res] MSA mask
            chunk_size:
                Size of chunks into which the inputs are split along their
                batch dimensions. A low value decreases memory overhead at the 
                cost of slower execution. Chunking is not performed by default.
        """ 
        # [*, N_res, N_seq, C_in]
        m = m.transpose(-2, -3)

        m = self._msa_att(m, chunk_size=chunk_size)

        # [*, N_seq, N_res, C_in]
        m = m.transpose(-2, -3)

        return m


class MSAColumnGlobalAttention(nn.Module):
    def __init__(
        self, c_in, c_hidden, no_heads, inf=1e9, eps=1e-10,
    ):
        super(MSAColumnGlobalAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.eps = eps

        self.layer_norm_m = nn.LayerNorm(c_in)

        self.global_attention = GlobalAttention(
            c_in=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
            inf=inf,
            eps=eps,
        )

    @torch.jit.ignore
    def _chunk(self,
        m: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        mha_input = {
            "m": m,
        }
        return chunk_layer(
            self.global_attention,
            mha_input,
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2]),
        )

    def forward(
        self, 
        m: torch.Tensor, 
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        n_seq, n_res, c_in = m.shape[-3:]

        # [*, N_res, N_seq, C_in]
        m = m.transpose(-2, -3)

        # [*, N_res, N_seq, C_in]
        m = self.layer_norm_m(m)

        if chunk_size is not None:
            m = self._chunk(m, chunk_size) 
        else:
            m = self.global_attention(m=m)

        # [*, N_seq, N_res, C_in]
        m = m.transpose(-2, -3)

        return m

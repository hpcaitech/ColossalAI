from typing import Optional, Tuple

import einops
import torch
import torch.nn as nn
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoder

from .attention import lower_triangular_attention


class XOPTAttention(OPTAttention):

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        assert is_cross_attention is False
        assert past_key_value is None
        assert layer_head_mask is None
        # assert output_attentions is False

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states).view(bsz, tgt_len, self.num_heads, self.head_dim)
        # get key, value proj
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        src_len = key_states.size(1)
        assert tgt_len == src_len

        attn_output = lower_triangular_attention(query=query_states, key=key_states, value=value_states, p=self.dropout)

        if attn_output.size() != (bsz, tgt_len, self.num_heads, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {(bsz, tgt_len, self.num_heads, self.head_dim)}, but is'
                             f' {attn_output.size()}')

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value


class XOPTDecoder(OPTDecoder):

    def forward(self, *args, **kwargs):
        assert 'attention_mask' in kwargs, 'please pass attention_mask as a kwarg'
        attn_mask = kwargs.get('attention_mask')
        # assert torch.all(attn_mask == 1), 'only accept no padding mask'

        head_mask = kwargs.get('head_mask', None)
        assert head_mask is None, 'head mask should be None'

        output_attn = kwargs.get('output_attentions', False)
        if output_attn:
            Warning('output_attentions is not supported for XOPTDecoder')

        return super().forward(*args, **kwargs)

from typing import Optional, Tuple

import torch
import xformers.ops as xops
from torch import Tensor
from transformers.models.opt.modeling_opt import OPTAttention


# This is modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py
class XOPTAttention(OPTAttention):
    # def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
    #     return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).contiguous()

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        past_key_value: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        layer_head_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        if not self.training:
            return super().forward(hidden_states, key_value_states, past_key_value, attention_mask, layer_head_mask,
                                   output_attentions)
        """Input shape: Batch x Time x Channel"""
        assert layer_head_mask is None, 'Xformers attention does not support layer_head_mask'
        assert not output_attentions, 'Xformers attention does not support output_attentions'

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
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

        query_states = self._shape(query_states, tgt_len, bsz).transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = xops.memory_efficient_attention(query_states,
                                                      key_states,
                                                      value_states,
                                                      attn_bias=xops.LowerTriangularMask(),
                                                      p=self.dropout if self.training else 0.0,
                                                      scale=self.scaling)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        attn_weights_reshaped = None

        return attn_output, attn_weights_reshaped, past_key_value

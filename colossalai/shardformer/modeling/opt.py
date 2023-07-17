from typing import Optional, Tuple

import torch

__all__ = ['get_opt_forward']


def get_opt_forward():

    from colossalai.kernel.cuda_native.flash_attention import AttnMaskType, ColoAttention

    def opt_flash_attention_forward(
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
        bsz, tgt_len, _ = hidden_states.size()
        assert tgt_len % 4 == 0, "Flash Attention Error: The sequence length should be a multiple of 4."

        attention_input_shape = (bsz, -1, self.num_heads, self.head_dim)
        # get query proj
        # query_states = self._shape(self.q_proj(hidden_states), -1, bsz)
        query_states = self.q_proj(hidden_states).view(*attention_input_shape)
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k, v, cross_attentions
            key_states = past_key_value[0].transpose(1, 2).contiguous().view(*attention_input_shape)
            value_states = past_key_value[1].transpose(1, 2).contiguous().view(*attention_input_shape)
        elif is_cross_attention:
            # cross_attentions
            key_states = self.k_proj(key_value_states).view(*attention_input_shape)
            value_states = self.v_proj(key_value_states).view(*attention_input_shape)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self.k_proj(hidden_states).view(*attention_input_shape)
            value_states = self.v_proj(hidden_states).view(*attention_input_shape)
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
        else:
            # self_attention
            key_states = self.k_proj(hidden_states).view(*attention_input_shape)
            value_states = self.v_proj(hidden_states).view(*attention_input_shape)

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
        if layer_head_mask != None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                                 f" {layer_head_mask.size()}")
        flash_attention_mask = None
        attn_mask_type = AttnMaskType.causal
        if attention_mask != None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}")
            flash_attention_mask = ~(attention_mask[:, :, -1].squeeze(1).to(torch.bool)).contiguous()
            attn_mask_type = AttnMaskType.paddedcausal

        attention = ColoAttention(embed_dim=self.embed_dim,
                                  num_heads=self.num_heads,
                                  dropout=self.dropout,
                                  scale=self.scaling)
        attn_output = attention(query_states,
                                key_states,
                                value_states,
                                attn_mask=flash_attention_mask,
                                attn_mask_type=attn_mask_type)

        attn_output = self.out_proj(attn_output)
        return attn_output, None, past_key_value

    return opt_flash_attention_forward

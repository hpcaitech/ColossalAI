from typing import Optional, Tuple, Union

import torch

__all__ = ['get_gpt2_forward']


def get_gpt2_forward():

    from colossalai.kernel.cuda_native.flash_attention import AttnMaskType, ColoAttention

    def gpt2_flash_attention_forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        _, tgt_len, _ = hidden_states.size()
        assert tgt_len % 4 == 0, "Flash Attention Error: The sequence length should be a multiple of 4."

        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`.")

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = split_heads(query, self.num_heads, self.head_dim)
        key = split_heads(key, self.num_heads, self.head_dim)
        value = split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if not self.is_cross_attention:
            attn_mask_type = AttnMaskType.causal
            flash_attention_mask = None
        if attention_mask != None:
            if attn_mask_type == AttnMaskType.causal:
                attn_mask_type == AttnMaskType.paddedcausal
            else:
                attn_mask_type = AttnMaskType.padding
            flash_attention_mask = ~(attention_mask[:, :, -1].squeeze(1).to(torch.bool)).contiguous()

        scale = value.size(-1)**-0.5
        if self.scale_attn_by_inverse_layer_idx:
            scale = scale * (1 / float(self.layer_idx + 1))

        # use coloattention
        attention = ColoAttention(embed_dim=self.embed_dim,
                                  num_heads=self.num_heads,
                                  dropout=self.attn_dropout.p,
                                  scale=scale)

        attn_output = attention(query, key, value, attn_mask=flash_attention_mask, attn_mask_type=attn_mask_type)

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output, present, None)

        return outputs

    return gpt2_flash_attention_forward


def split_heads(tensor, num_heads, attn_head_size):
    """
    Splits hidden_size dim into attn_head_size and num_heads
    """
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    return tensor

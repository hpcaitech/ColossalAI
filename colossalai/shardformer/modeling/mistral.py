from typing import Optional, Tuple

import torch


def get_mistral_flash_attention_forward():
    from transformers.models.mistral.modeling_mistral import MistralAttention, apply_rotary_pos_emb, repeat_kv

    from colossalai.kernel.cuda_native import AttnMaskType, ColoAttention

    def forward(
        self: MistralAttention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        assert q_len % 4 == 0, "Flash Attention Error: The sequence length should be a multiple of 4."

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = (
            self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        me_input_shape = (bsz, q_len, self.num_heads, self.head_dim)
        query_states = query_states.transpose(1, 2).contiguous().view(*me_input_shape)
        key_states = key_states.transpose(1, 2).contiguous().view(*me_input_shape)
        value_states = value_states.transpose(1, 2).contiguous().view(*me_input_shape)

        flash_attention_mask = None
        attn_mask_type = AttnMaskType.causal
        if attention_mask != None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            flash_attention_mask = ~(attention_mask[:, :, -1].squeeze(1).to(torch.bool)).contiguous()
            attn_mask_type = AttnMaskType.paddedcausal

        attention = ColoAttention(embed_dim=self.hidden_size, num_heads=self.num_heads)
        attn_output = attention(
            query_states, key_states, value_states, attn_mask=flash_attention_mask, attn_mask_type=attn_mask_type
        )

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    return forward

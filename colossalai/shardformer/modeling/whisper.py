from typing import Optional, Tuple

import torch
from torch import nn


def get_whisper_flash_attention_forward():

    from transformers.models.whisper.modeling_whisper import WhisperAttention

    from colossalai.kernel.cuda_native.flash_attention import AttnMaskType, ColoAttention

    def forward(
        self: WhisperAttention,
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

        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (is_cross_attention and past_key_value is not None
                and past_key_value[0].shape[1] == key_value_states.shape[1]):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = shape(self.k_proj(key_value_states), -1, bsz, self.num_heads, self.head_dim)
            value_states = shape(self.v_proj(key_value_states), -1, bsz, self.num_heads, self.head_dim)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = shape(self.k_proj(hidden_states), -1, bsz, self.num_heads, self.head_dim)
            value_states = shape(self.v_proj(hidden_states), -1, bsz, self.num_heads, self.head_dim)
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
        else:
            # self_attention
            key_states = shape(self.k_proj(hidden_states), -1, bsz, self.num_heads, self.head_dim)
            value_states = shape(self.v_proj(hidden_states), -1, bsz, self.num_heads, self.head_dim)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        # get query proj
        query_states = shape(self.q_proj(hidden_states), tgt_len, bsz, self.num_heads, self.head_dim)

        src_len = key_states.size(1)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                                 f" {layer_head_mask.size()}")

        attn_type = None
        flash_attention_mask = None

        if self.is_decoder:
            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                    )
                flash_attention_mask = ~(attention_mask[:, :, -1].squeeze(1).to(torch.bool).contiguous())
                attn_type = AttnMaskType.paddedcausal

        attention = ColoAttention(embed_dim=self.embed_dim,
                                  num_heads=self.num_heads,
                                  dropout=self.dropout,
                                  scale=self.scaling)
        attn_output = attention(query_states,
                                key_states,
                                value_states,
                                attn_mask=flash_attention_mask,
                                attn_mask_type=attn_type)

        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value

    return forward


def shape(tensor: torch.Tensor, seq_len: int, bsz: int, num_heads: int, head_dim: int):
    return tensor.view(bsz, seq_len, num_heads, head_dim).contiguous()

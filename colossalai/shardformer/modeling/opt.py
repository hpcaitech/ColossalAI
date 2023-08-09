from typing import Optional, Tuple

import torch
from torch import nn


def get_opt_flash_attention_forward():

    from transformers.models.opt.modeling_opt import OPTAttention

    from colossalai.kernel.cuda_native import AttnMaskType, ColoAttention

    def forward(
        self: OPTAttention,
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

    return forward


def get_jit_fused_opt_decoder_layer_forward():

    from transformers.models.opt.modeling_opt import OPTDecoderLayer

    def forward(
        self: OPTDecoderLayer,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        hidden_states = self.dropout_add(hidden_states, residual, self.dropout, self.training)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)

        hidden_states = self.dropout_add(hidden_states, residual, self.dropout, self.training).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    return forward

import torch
import torch.nn.functional as F


def get_flash_core_attention_forward():

    from colossalai.kernel.cuda_native.flash_attention import AttnMaskType, ColoAttention
    from tests.kit.model_zoo.transformers.chatglm2_6b.modeling_chatglm import CoreAttention

    def forward(self: CoreAttention, query_layer, key_layer, value_layer, attention_mask):
        pytorch_major_version = int(torch.__version__.split(".")[0])
        if pytorch_major_version >= 2:
            query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]
            if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer,
                                                                                 key_layer,
                                                                                 value_layer,
                                                                                 is_causal=True)
            else:
                if attention_mask is not None:
                    attention_mask = ~attention_mask
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                 attention_mask)
            context_layer = context_layer.permute(2, 0, 1, 3)
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.reshape(*new_context_layer_shape)
        else:
            # Raw attention scores
            query_layer = query_layer.permute(1, 0, 2, 3).contiguous()
            key_layer = key_layer.permute(1, 0, 2, 3).contiguous()
            value_layer = value_layer.permute(1, 0, 2, 3).contiguous()

            scale = 1.0 / self.norm_factor
            if self.coeff is not None:
                scale = scale * self.coeff

            flash_attention_mask = None
            attn_mask_type = None
            if (attention_mask is None):
                attn_mask_type = AttnMaskType.causal
            elif attention_mask is not None:
                flash_attention_mask = ~(attention_mask[:, :, -1].squeeze(1).to(torch.bool)).contiguous()
                attn_mask_type = AttnMaskType.paddedcausal

            attention = ColoAttention(embed_dim=self.hidden_size_per_partition,
                                      num_heads=self.num_attention_heads_per_partition,
                                      dropout=self.attention_dropout.p,
                                      scale=scale)
            context_layer = attention(query_layer,
                                      key_layer,
                                      value_layer,
                                      attn_mask=flash_attention_mask,
                                      attn_mask_type=attn_mask_type)

            context_layer = context_layer.permute(1, 0, -1).contiguous()

        return context_layer

    return forward


def get_jit_fused_glm_block_forward():

    from tests.kit.model_zoo.transformers.chatglm2_6b.modeling_chatglm import GLMBlock

    def forward(
        self: GLMBlock,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache=None,
        use_cache=True,
    ):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, kv_cache = self.self_attention(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = self.dropout_add(attention_output, residual, self.hidden_dropout, self.training)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = self.dropout_add(mlp_output, residual, self.hidden_dropout, self.training)

        return output, kv_cache

    return forward

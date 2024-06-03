import torch
import torch.nn as nn

from colossalai.kernel.jit import bias_dropout_add_fused_inference, bias_dropout_add_fused_train
from colossalai.legacy.nn.layer.parallel_sequence import TransformerSelfAttentionRing
from colossalai.nn.layer.layernorm import MixedFusedLayerNorm as LayerNorm

from .dropout import get_bias_dropout_add
from .mlp import TransformerMLP


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


class BertLayer(nn.Module):
    """A single transformer layer.
    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        layer_number,
        hidden_size,
        num_attention_heads,
        attention_dropout,
        mlp_ratio,
        hidden_dropout,
        is_naive_fp16,
        apply_residual_connection_post_layernorm=False,
        fp32_residual_connection=False,
        bias_dropout_fusion: bool = True,
        convert_fp16_to_fp32_in_softmax: bool = False,
    ):
        super().__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.fp32_residual_connection = fp32_residual_connection

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(hidden_size)

        # Self attention.
        self.self_attention = TransformerSelfAttentionRing(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            attention_mask_func=attention_mask_func,
            layer_number=layer_number,
            apply_query_key_layer_scaling=True,
            convert_fp16_to_fp32_in_softmax=convert_fp16_to_fp32_in_softmax,
            fp16=is_naive_fp16,
        )

        self.hidden_dropout = hidden_dropout
        self.bias_dropout_fusion = bias_dropout_fusion

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNorm(hidden_size)

        self.mlp = TransformerMLP(hidden_size=hidden_size, mlp_ratio=mlp_ratio)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, sub_seq_len, hidden_size]
        # attention_mask: [batch_size, 1, sub_seq_len, seq_len]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output, attention_bias = self.self_attention(layernorm_output, attention_mask)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # jit scripting for a nn.module (with dropout) is not
        # trigerring the fusion kernel. For now, we use two
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output, attention_bias.expand_as(residual), residual, self.hidden_dropout
            )

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            output = bias_dropout_add_func(mlp_output, mlp_bias.expand_as(residual), residual, self.hidden_dropout)

        return output

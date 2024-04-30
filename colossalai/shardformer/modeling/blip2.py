from typing import Optional, Tuple

import torch
import torch.nn as nn

from colossalai.shardformer.layer import ColoAttention


def forward_fn():
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        mixed_qkv = self.qkv(hidden_states)

        # modified from original code, which is:
        # mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(
        #     2, 0, 3, 1, 4
        # )
        # to:
        mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        query_states, key_states, value_states = (
            mixed_qkv[0],
            mixed_qkv[1],
            mixed_qkv[2],
        )

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

        attention_scores = attention_scores * self.scale

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3)

        new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        output = self.projection(context_layer)

        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs

    return forward


def get_blip2_flash_attention_forward():
    from transformers.models.blip_2.modeling_blip_2 import Blip2Attention

    def forward(
        self: Blip2Attention,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        assert head_mask is None, "head_mask is not supported in FlashAttention"
        bsz, tgt_len, embed_dim = hidden_states.size()
        mixed_qkv = self.qkv(hidden_states)
        mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query_states, key_states, value_states = (
            mixed_qkv[0],
            mixed_qkv[1],
            mixed_qkv[2],
        )

        dropout_p = self.dropout.p if self.training else 0.0
        context_layer = ColoAttention.attention(
            query_states,
            key_states,
            value_states,
            dropout_p=dropout_p,
            scale=self.scale,
        )
        context_layer = context_layer.permute(0, 2, 1, 3).reshape(bsz, tgt_len, self.embed_dim)

        output = self.projection(context_layer)
        outputs = (output, None)

        return outputs

    return forward


def get_jit_fused_blip2_QFormer_self_output_forward():
    from transformers.models.blip_2.modeling_blip_2 import Blip2QFormerSelfOutput

    def forward(
        self: Blip2QFormerSelfOutput,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout_add(hidden_states, input_tensor, self.dropout.p, self.dropout.training)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

    return forward


def get_jit_fused_blip2_QFormer_output_forward():
    from transformers.models.blip_2.modeling_blip_2 import Blip2QFormerOutput

    def forward(
        self: Blip2QFormerOutput,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout_add(hidden_states, input_tensor, self.dropout.p, self.dropout.training)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

    return forward


def get_jit_fused_blip2_mlp_forward():
    from transformers.models.blip_2.modeling_blip_2 import Blip2MLP

    from colossalai.kernel.jit.bias_gelu import GeLUFunction as JitGeLUFunction

    def forward(self: Blip2MLP, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, bias = self.fc1(hidden_states)
        hidden_states = JitGeLUFunction.apply(hidden_states, bias)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

    return forward

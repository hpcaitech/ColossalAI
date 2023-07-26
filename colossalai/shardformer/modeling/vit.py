import math
from typing import Optional, Tuple, Union

import torch
from torch import nn


def get_vit_flash_self_attention_forward():

    from transformers.models.vit.modeling_vit import ViTSelfAttention

    from colossalai.kernel.cuda_native.flash_attention import ColoAttention

    def transpose_for_scores(x: torch.Tensor, num_attention_heads, attention_head_size) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = x.view(new_x_shape)
        return x

    def forward(self: ViTSelfAttention,
                hidden_states: torch.Tensor,
                head_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = transpose_for_scores(self.key(hidden_states), self.num_attention_heads, self.attention_head_size)
        value_layer = transpose_for_scores(self.value(hidden_states), self.num_attention_heads,
                                           self.attention_head_size)
        query_layer = transpose_for_scores(mixed_query_layer, self.num_attention_heads, self.attention_head_size)

        scale = 1.0 / math.sqrt(self.attention_head_size)
        attention = ColoAttention(embed_dim=self.all_head_size,
                                  num_heads=self.num_attention_heads,
                                  dropout=self.dropout.p,
                                  scale=scale)
        context_layer = attention(query_layer, key_layer, value_layer)

        outputs = (context_layer,)

        return outputs

    return forward


def get_jit_fused_vit_output_forward():

    from transformers.models.vit.modeling_vit import ViTOutput

    def forward(self: ViTOutput, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout_add(hidden_states, input_tensor, self.dropout.p, self.dropout.training)
        return hidden_states

    return forward

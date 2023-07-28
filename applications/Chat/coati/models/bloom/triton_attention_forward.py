from typing import Optional, Tuple

import torch
from torch.nn import functional as F
from transformers.models.bloom.configuration_bloom import BloomConfig
from transformers.models.bloom.modeling_bloom import BloomAttention

from colossalai.kernel.triton.ops import compute_attention_for_bloom


def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *required*):
            esidual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


class TritonBloomAttention(BloomAttention):

    def __init__(self, config: BloomConfig):
        super(TritonBloomAttention, self).__init__(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        fused_qkv = self.query_key_value(hidden_states)    # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
        q_length = query_layer.shape[1]
        batch_size = query_layer.shape[0]
        num_heads = query_layer.shape[2]

        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=2)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        _, _, kv_length = key_layer.shape
        alibi = alibi.view(batch_size, num_heads, q_length, -1)

        context_layer = compute_attention_for_bloom(
            q=query_layer.view(batch_size, self.num_heads, q_length, self.head_dim),
            k=key_layer.view(batch_size, self.num_heads, self.head_dim, kv_length),
            v=value_layer.view(batch_size, self.num_heads, kv_length, self.head_dim),
            alibi=alibi,
            beta=self.beta,
            scale=self.inv_norm_factor,
            attention_mask=attention_mask,
            drop_out=self.hidden_dropout,
            head_mask=head_mask,
            layer_past=layer_past,
            use_cache=True,
        )

        if use_cache:
            present = (key_layer, value_layer)
        else:
            present = None

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices):int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices):int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs

from typing import Optional, Tuple

import torch
from torch.nn import functional as F
from transformers.models.bloom.configuration_bloom import BloomConfig
from transformers.models.bloom.modeling_bloom import BloomAttention

#import cuda kernels 
from col_fused_softmax_lib import scaled_masked_softmax_forward
from col_linear_lib import dense_layer_fp32_forward, dense_layer_fp16_forward, batch_dense_layer_fp16_forward


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

def softmax(data, dim = -1):
    """
        data must be 4-dimensional, and it only supports float16 dtype
    """
    batch, _, M, N = data.shape
    
    mask = torch.zeros(size = (batch, 1, M, N), device=data.get_device(), dtype=torch.uint8)
    res = scaled_masked_softmax_forward(data, mask, 1)
    return res

def softmax_with_mask(data, mask):
    """
        data must be 4-dimensional, and it only supports float16 dtype,
        mask must be (b, 1, M, N) shape
    """
    return scaled_masked_softmax_forward(data, mask, 1)

def batch_linear(data, weight, alibi = None, alpha = 1, beta = 0):
    """
      it is equivalent to alibi.bmm(data, weight)
      only supports float16
    """
    batch_count, M, K = data.shape
    _, _, N = weight.shape
    assert batch_count == weight.shape[0], "the batch size must be matched"
    if alibi is None:
        out = torch.empty((batch_count, M, N), dtype=torch.float16, device=data.get_device())
    else:
        out = alibi

    batch_dense_layer_fp16_forward(data, weight, out, alpha, beta, 99)
    return out


class CudaBloomAttention(BloomAttention):

    def __init__(self, config: BloomConfig):
        super(CudaBloomAttention, self).__init__(config)

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
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, q_length, _, _ = query_layer.shape

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

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        alibi = alibi.expand((batch_size * self.num_heads, q_length, kv_length))
        matmul_result = batch_linear(query_layer, key_layer, alibi, alpha = self.inv_norm_factor, beta = self.beta)

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)

        # TODO: need to fix it to make it suitable for general cases.
        if q_length%8 == 0:
            attention_mask = attention_mask.to(dtype=torch.uint8)
            attention_probs = softmax_with_mask(attention_scores, attention_mask)
        else:
            attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
            attention_probs = F.softmax(attn_weights, dim=-1)

        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        # context_layer = torch.bmm(attention_probs_reshaped, value_layer)
        context_layer = batch_linear(attention_probs_reshaped, value_layer, None, alpha = 1, beta = 0)


        # change view [batch_size, q_length, num_heads * head_dim]
        context_layer = self._merge_heads(context_layer)

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs
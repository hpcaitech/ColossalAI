from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.nn import functional as F


def build_bloom_alibi_tensor_fn(process_group: ProcessGroup) -> torch.Tensor:

    def build_bloom_alibi_tensor(self, attention_mask: torch.Tensor, num_heads: int,
                                 dtype: torch.dtype) -> torch.Tensor:
        """
        Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
        relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
        `softmax(l+a) = softmax(l)`. Based on
        https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
        TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

        Args:
        Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
            attention_mask (`torch.Tensor`):
                Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
            num_heads (`int`, *required*):
                number of heads
            dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
                dtype of the output tensor
        """
        import math

        if dist.is_initialized():
            world_size = dist.get_world_size(process_group)
            num_heads = num_heads * world_size

        batch_size, seq_length = attention_mask.shape
        closest_power_of_2 = 2**math.floor(math.log2(num_heads))
        base = torch.tensor(2**(-(2**-(math.log2(closest_power_of_2) - 3))),
                            device=attention_mask.device,
                            dtype=torch.float32)
        powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.pow(base, powers)

        if closest_power_of_2 != num_heads:
            extra_base = torch.tensor(2**(-(2**-(math.log2(2 * closest_power_of_2) - 3))),
                                      device=attention_mask.device,
                                      dtype=torch.float32)
            num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
            extra_powers = torch.arange(1,
                                        1 + 2 * num_remaining_heads,
                                        2,
                                        device=attention_mask.device,
                                        dtype=torch.int32)
            slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

        # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
        # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
        # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
        # => the query_length dimension will then be broadcasted correctly
        # This is more or less identical to T5's relative position bias:
        # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
        arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
        alibi = slopes[..., None] * arange_tensor
        if dist.is_initialized():
            num_heads_per_rank = int(num_heads / dist.get_world_size(process_group))
            offset = dist.get_rank(process_group) * num_heads_per_rank
            alibi = alibi.view(batch_size, num_heads, 1, seq_length)
            alibi = alibi[:, offset:num_heads_per_rank + offset, :, :]
            return alibi.reshape(batch_size * num_heads_per_rank, 1, seq_length).to(dtype)
        else:
            return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)

    return build_bloom_alibi_tensor

def get_bloom_forward():

    try:
        from xformers.ops import memory_efficient_attention as me_attention
    except:
        raise ImportError("Error: xformers module is not installed. Please install it to use flash attention.")
    from transformers.models.bloom.modeling_bloom import dropout_add

    def bloom_flash_attention_forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,):

        fused_qkv = self.query_key_value(hidden_states)
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
        batch_size, tgt_len, _ = hidden_states.size()
        assert tgt_len % 4 == 0, "Flash Attention Error: The sequence length should be a multiple of 4."

        _, kv_length, _, _ = key_layer.size()

        proj_shape = (batch_size, tgt_len, self.num_heads, self.head_dim)
        query_layer = query_layer.contiguous().view(*proj_shape)
        key_layer = key_layer.contiguous().view(*proj_shape)
        value_layer = value_layer.contiguous().view(*proj_shape)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=1)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None
        
        tgt_len = key_layer.size()[1]

        attention_numerical_mask = torch.zeros((batch_size, self.num_heads, tgt_len, kv_length), dtype=torch.float32, device=query_layer.device, requires_grad=True)
        attention_numerical_mask = attention_numerical_mask + alibi.view(batch_size, self.num_heads, 1, kv_length) * self.beta
        attention_numerical_mask = torch.masked_fill(attention_numerical_mask, attention_mask, torch.finfo(torch.float32).min)
        
        context_layer = me_attention(query_layer, key_layer, value_layer, attn_bias=attention_numerical_mask, scale=self.inv_norm_factor, p=self.attention_dropout.p)
        context_layer = context_layer.reshape(-1, kv_length, self.hidden_size)
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

        # TODO to replace with the bias_dropout_add function in jit
        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)
        outputs = (output_tensor, present, None)

        return outputs
        
    return bloom_flash_attention_forward
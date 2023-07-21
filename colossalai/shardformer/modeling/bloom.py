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


def get_bloom_flash_attention_forward(enabel_jit_fused=False):

    try:
        from xformers.ops import memory_efficient_attention as me_attention
    except:
        raise ImportError("Error: xformers module is not installed. Please install it to use flash attention.")
    from transformers.models.bloom.modeling_bloom import BloomAttention

    def forward(
        self: BloomAttention,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):

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

        attention_numerical_mask = torch.zeros((batch_size, self.num_heads, tgt_len, kv_length),
                                               dtype=torch.float32,
                                               device=query_layer.device,
                                               requires_grad=True)
        attention_numerical_mask = attention_numerical_mask + alibi.view(batch_size, self.num_heads, 1,
                                                                         kv_length) * self.beta
        attention_numerical_mask = torch.masked_fill(attention_numerical_mask, attention_mask,
                                                     torch.finfo(torch.float32).min)

        context_layer = me_attention(query_layer,
                                     key_layer,
                                     value_layer,
                                     attn_bias=attention_numerical_mask,
                                     scale=self.inv_norm_factor,
                                     p=self.attention_dropout.p)
        context_layer = context_layer.reshape(-1, kv_length, self.hidden_size)
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

        # TODO to replace with the bias_dropout_add function in jit
        output_tensor = self.dropout_add(output_tensor, residual, self.hidden_dropout, self.training)
        outputs = (output_tensor, present, None)

        return outputs

    return forward


def get_jit_fused_bloom_attention_forward():

    from transformers.models.bloom.modeling_bloom import BloomAttention

    def forward(
        self: BloomAttention,
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

        # [batch_size * num_heads, q_length, kv_length]
        # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
        matmul_result = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16:
            attention_scores = attention_scores.to(torch.float)
        attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)

        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = self._merge_heads(context_layer)

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

        output_tensor = self.dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs

    return forward


def get_jit_fused_bloom_mlp_forward():

    from transformers.models.bloom.modeling_bloom import BloomMLP

    def forward(self: BloomMLP, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

        if self.pretraining_tp > 1 and self.slow_but_exact:
            intermediate_output = torch.zeros_like(residual)
            slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
            for i in range(self.pretraining_tp):
                intermediate_output = intermediate_output + F.linear(
                    hidden_states[:, :, int(i * slices):int((i + 1) * slices)],
                    self.dense_4h_to_h.weight[:, int(i * slices):int((i + 1) * slices)],
                )
        else:
            intermediate_output = self.dense_4h_to_h(hidden_states)
        output = self.dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)
        return output

    return forward


def get_jit_fused_bloom_gelu_forward():

    from transformers.models.bloom.modeling_bloom import BloomGelu

    from colossalai.kernel.jit.bias_gelu import GeLUFunction as JitGeLUFunction

    def forward(self: BloomGelu, x: torch.Tensor) -> torch.Tensor:
        bias = torch.zeros_like(x)
        if self.training:
            return JitGeLUFunction.apply(x, bias)
        else:
            return self.bloom_gelu_forward(x, bias)

    return forward

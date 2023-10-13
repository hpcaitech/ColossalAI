
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F

def build_falcon_alibi_tensor_fn(process_group: ProcessGroup) -> torch.Tensor:
    def build_falcon_alibi_tensor(
        self, attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype
    ) -> torch.Tensor:
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
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        base = torch.tensor(
            2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.pow(base, powers)

        if closest_power_of_2 != num_heads:
            extra_base = torch.tensor(
                2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
                device=attention_mask.device,
                dtype=torch.float32,
            )
            num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
            extra_powers = torch.arange(
                1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32
            )
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
            alibi = alibi[:, offset : num_heads_per_rank + offset, :, :]
            return alibi.reshape(batch_size * num_heads_per_rank, 1, seq_length).to(dtype)
        else:
            return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)

    return build_falcon_alibi_tensor


def get_tp_falcon_decoder_layer_forward():
    from transformers.models.falcon.modeling_falcon import FalconDecoderLayer, dropout_add
    
    def forward(
        self: FalconDecoderLayer,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        residual = hidden_states

        if self.config.new_decoder_architecture:
            attention_layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)
        else:
            attention_layernorm_out = self.input_layernorm(hidden_states)

        # Self attention.
        attn_outputs = self.self_attention(
            attention_layernorm_out,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        if not self.config.new_decoder_architecture:
            if self.config.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                residual = dropout_add(
                    attention_output, residual, self.config.attention_dropout, training=self.training
                )
                mlp_layernorm_out = self.post_attention_layernorm(residual)

        outputs = attn_outputs[1:]

        # MLP.
        mlp_output = self.mlp(mlp_layernorm_out)

        if self.config.new_decoder_architecture or self.config.parallel_attn:
            mlp_output = mlp_output + attention_output

        output = dropout_add(mlp_output, residual, self.config.hidden_dropout, training=self.training)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions
    
    return forward

def get_falcon_flash_attention_forward():
    try:
        from xformers.ops import memory_efficient_attention as me_attention
    except:
        raise ImportError("Error: xformers module is not installed. Please install it to use flash attention.")
    from transformers.models.falcon.modeling_falcon import FalconAttention
    from colossalai.kernel.cuda_native import AttnMaskType, ColoAttention

    def forward(
        self: FalconAttention,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, query_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, query_length, self.head_dim)
        key_layer = key_layer.transpose(1, 2).reshape(
            batch_size * num_kv_heads,
            query_length,
            self.head_dim,
        )
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * num_kv_heads, query_length, self.head_dim)


        past_kv_length = 0 if layer_past is None else layer_past[0].shape[1]
        query_layer, key_layer = self.maybe_rotary(query_layer, key_layer, past_kv_length)


        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, kv_length, head_dim]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=1)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        _, kv_length, _ = key_layer.shape
        if use_cache:
            present = (key_layer, value_layer)
        else:
            present = None

        attention_mask_float = (attention_mask * 1.0).masked_fill(attention_mask, float("-1e9")).to(query_layer.dtype)

        query_layer_ = query_layer.reshape(batch_size, self.num_heads, -1, self.head_dim).transpose(1, 2).contiguous()
        key_layer_ = key_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim).transpose(1, 2).contiguous()
        value_layer_ = value_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim).transpose(1, 2).contiguous()

        if alibi is not None:
            attention_mask_float = (
                attention_mask_float + alibi.view(batch_size, self.num_heads, 1, kv_length) * self.beta
            )
 
        batch_size, src_len = query_layer_.size()[0], query_layer_.size()[1]
        tgt_len = key_layer_.size()[1]
        attention_mask_float = attention_mask_float.expand(
            batch_size, self.num_heads, src_len, tgt_len
        ).contiguous()
        context_layer = me_attention(
            query_layer_,
            key_layer_,
            value_layer_,
            attn_bias=attention_mask_float,
            scale=self.inv_norm_factor,
            p=self.attention_dropout.p,
        )
        batch_size, seq_length, _, _ = context_layer.shape
        context_layer = context_layer.reshape(batch_size, seq_length, -1)
        
        output_tensor = self.dense(context_layer)

        return output_tensor, present

    return forward

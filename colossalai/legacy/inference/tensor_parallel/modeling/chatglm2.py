import os
from typing import Optional, Tuple

import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from colossalai.inference.tensor_parallel.batch_infer_state import BatchInferState
from colossalai.kernel.triton.token_attention_kernel import Llama2TokenAttentionForwards
from colossalai.shardformer.modeling.chatglm2_6b.modeling_chatglm import (
    ChatGLMForConditionalGeneration,
    ChatGLMModel,
    GLMBlock,
    GLMTransformer,
    SelfAttention,
    split_tensor_along_last_dim,
)

from ._utils import copy_kv_to_mem_cache

try:
    from lightllm.models.chatglm2.triton_kernel.rotary_emb import rotary_emb_fwd as chatglm2_rotary_emb_fwd
    from lightllm.models.llama2.triton_kernel.context_flashattention_nopad import (
        context_attention_fwd as lightllm_llama2_context_attention_fwd,
    )

    HAS_LIGHTLLM_KERNEL = True
except:
    print("please install lightllm from source to run inference: https://github.com/ModelTC/lightllm")
    HAS_LIGHTLLM_KERNEL = False


# This func is same as Llama model init_to_get_rotary, we should move them into _utils.py
def _init_to_get_rotary(self, base=10000):
    self.config.head_dim_ = self.config.hidden_size // self.config.num_attention_heads
    if not hasattr(self.config, "rope_scaling"):
        rope_scaling_factor = 1.0
    else:
        rope_scaling_factor = self.config.rope_scaling.factor if self.config.rope_scaling is not None else 1.0
    if hasattr(self.config, "max_sequence_length"):
        max_seq_len = self.config.max_sequence_length
    elif hasattr(self.config, "max_position_embeddings"):
        max_seq_len = self.config.max_position_embeddings * rope_scaling_factor
    else:
        max_seq_len = 2048 * rope_scaling_factor
    base = float(base)

    # NTK  ref: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    try:
        ntk_alpha = float(os.environ.get("INFER_NTK_ALPHA", 1))
        assert ntk_alpha >= 1
        if ntk_alpha > 1:
            print(f"Note: NTK enabled, alpha set to {ntk_alpha}")
        max_seq_len *= ntk_alpha
        base = base * (ntk_alpha ** (self.head_dim_ / (self.head_dim_ - 2)))  # Base change formula
    except:
        pass
    n_elem = self.config.head_dim_ // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, n_elem, 2, device="cpu", dtype=torch.float32) / n_elem))
    t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32) / rope_scaling_factor
    freqs = torch.outer(t, inv_freq)

    self._cos_cached = torch.cos(freqs).to(torch.float16).cuda()
    self._sin_cached = torch.sin(freqs).to(torch.float16).cuda()
    return


def get_masks(self, input_ids, past_length, padding_mask=None):
    batch_size, seq_length = input_ids.shape
    full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
    full_attention_mask.tril_()
    if past_length:
        full_attention_mask = torch.cat(
            (
                torch.ones(batch_size, seq_length, past_length, device=input_ids.device),
                full_attention_mask,
            ),
            dim=-1,
        )

    if padding_mask is not None:
        full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
    if not past_length and padding_mask is not None:
        full_attention_mask -= padding_mask.unsqueeze(-1) - 1
    full_attention_mask = (full_attention_mask < 0.5).bool()
    full_attention_mask.unsqueeze_(1)
    return full_attention_mask


class ChatGLM2InferenceForwards:
    """
    This class holds forwards for Chatglm2 inference.
    We intend to replace the forward methods for ChatGLMModel, ChatGLMEecoderLayer, and ChatGLMAttention.
    """

    @staticmethod
    def chatglm_for_conditional_generation_forward(
        self: ChatGLMForConditionalGeneration,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_last_logit: Optional[bool] = False,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        infer_state = self.infer_state

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if infer_state.is_context_stage:
            past_key_values_length = 0
        else:
            past_key_values_length = infer_state.max_len_in_batch - 1

        seq_length_with_past = seq_length + past_key_values_length

        # prefill stage at first
        if use_cache and seq_length != 1:
            infer_state.is_context_stage = True
            infer_state.context_mem_index = infer_state.cache_manager.alloc(infer_state.total_token_num)
            infer_state.init_block_loc(
                infer_state.block_loc, infer_state.seq_len, seq_length, infer_state.context_mem_index
            )
        else:
            infer_state.is_context_stage = False
            alloc_mem = infer_state.cache_manager.alloc_contiguous(batch_size)
            if alloc_mem is not None:
                infer_state.decode_is_contiguous = True
                infer_state.decode_mem_index = alloc_mem[0]
                infer_state.decode_mem_start = alloc_mem[1]
                infer_state.decode_mem_end = alloc_mem[2]
                infer_state.block_loc[:, seq_length_with_past - 1] = infer_state.decode_mem_index
            else:
                print(f" *** Encountered allocation non-contiguous")
                print(
                    f"    infer_state.cache_manager.past_key_values_length: {infer_state.cache_manager.past_key_values_length}"
                )
                infer_state.decode_is_contiguous = False
                alloc_mem = infer_state.cache_manager.alloc(batch_size)
                infer_state.decode_mem_index = alloc_mem
                # infer_state.decode_key_buffer = torch.empty((batch_size, self.tp_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
                # infer_state.decode_value_buffer = torch.empty((batch_size, self.tp_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
                infer_state.block_loc[:, seq_length_with_past - 1] = infer_state.decode_mem_index

        # related to rotary embedding
        if infer_state.is_context_stage:
            infer_state.position_cos = torch.index_select(self._cos_cached, 0, position_ids.view(-1)).view(
                position_ids.view(-1).shape[0], -1
            )
            infer_state.position_sin = torch.index_select(self._sin_cached, 0, position_ids.view(-1)).view(
                position_ids.view(-1).shape[0], -1
            )
        else:
            seq_len = infer_state.seq_len
            infer_state.position_cos = torch.index_select(self._cos_cached, 0, seq_len - 1).view(seq_len.shape[0], -1)
            infer_state.position_sin = torch.index_select(self._sin_cached, 0, seq_len - 1).view(seq_len.shape[0], -1)
            infer_state.other_kv_index = infer_state.block_loc[0, infer_state.max_len_in_batch - 1].item()

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            infer_state=infer_state,
        )

        hidden_states = transformer_outputs[0]
        if return_last_logit:
            hidden_states = hidden_states[-1:]
        lm_logits = self.transformer.output_layer(hidden_states)
        lm_logits = lm_logits.transpose(0, 1).contiguous()

        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def chatglm_model_forward(
        self: ChatGLMModel,
        input_ids,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        full_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        infer_state: BatchInferState = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(
                    batch_size=batch_size,
                    device=input_ids.device,
                    dtype=inputs_embeds.dtype,
                )
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        attention_mask.new_ones((batch_size, self.pre_seq_len)),
                        attention_mask,
                    ],
                    dim=-1,
                )
        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = get_masks(
                    self, input_ids, infer_state.cache_manager.past_key_values_length, padding_mask=attention_mask
                )

        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds,
            full_attention_mask,
            kv_caches=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            infer_state=infer_state,
        )

        # update indices
        # infer_state.block_loc[:, infer_state.max_len_in_batch-1] = infer_state.total_token_num + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        infer_state.start_loc = infer_state.start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        infer_state.seq_len += 1
        infer_state.max_len_in_batch += 1

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    @staticmethod
    def chatglm_encoder_forward(
        self: GLMTransformer,
        hidden_states,
        attention_mask,
        kv_caches=None,
        use_cache: Optional[bool] = True,
        output_hidden_states: Optional[bool] = False,
        infer_state: Optional[BatchInferState] = None,
    ):
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_layers)]
        presents = () if use_cache else None
        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None

        infer_state.decode_layer_id = 0
        for index in range(self.num_layers):
            layer = self.layers[index]

            layer_ret = layer(
                hidden_states,
                attention_mask,
                kv_cache=kv_caches[index],
                use_cache=use_cache,
                infer_state=infer_state,
            )

            infer_state.decode_layer_id += 1

            hidden_states, kv_cache = layer_ret
            if use_cache:
                presents = presents + (kv_cache,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer norm.
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, presents, all_hidden_states, all_self_attentions

    @staticmethod
    def chatglm_glmblock_forward(
        self: GLMBlock,
        hidden_states,
        attention_mask,
        kv_cache=None,
        use_cache=True,
        infer_state: Optional[BatchInferState] = None,
    ):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, kv_cache = self.self_attention(
            layernorm_output,
            attention_mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
            infer_state=infer_state,
        )
        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states
        layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = residual + layernorm_input
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output
        return output, kv_cache

    @staticmethod
    def chatglm_flash_attn_kvcache_forward(
        self: SelfAttention,
        hidden_states,
        attention_mask,
        kv_cache=None,
        use_cache=True,
        infer_state: Optional[BatchInferState] = None,
    ):
        assert use_cache is True, "use_cache should be set to True using this chatglm attention"
        # hidden_states: original :[sq, b, h] --> this [b, sq, h]
        batch_size = hidden_states.shape[0]
        hidden_size = hidden_states.shape[-1]
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(hidden_states)
        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1]
                + (
                    self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1]
                + (
                    self.num_multi_query_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (
                    self.num_multi_query_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )

        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
        cos, sin = infer_state.position_cos, infer_state.position_sin

        chatglm2_rotary_emb_fwd(
            query_layer.view(-1, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head), cos, sin
        )
        if self.multi_query_attention:
            chatglm2_rotary_emb_fwd(
                key_layer.view(-1, self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head),
                cos,
                sin,
            )
        else:
            chatglm2_rotary_emb_fwd(
                key_layer.view(-1, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head),
                cos,
                sin,
            )

        # reshape q k v  to [bsz*sql, num_heads, head_dim]   2*1 ,32/2 ,128
        query_layer = query_layer.reshape(
            -1, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head
        )
        key_layer = key_layer.reshape(
            -1, self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head
        )
        value_layer = value_layer.reshape(
            -1, self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head
        )

        if infer_state.is_context_stage:
            # first token generation:
            # copy key and value calculated in current step to memory manager
            copy_kv_to_mem_cache(
                infer_state.decode_layer_id,
                key_layer,
                value_layer,
                infer_state.context_mem_index,
                infer_state.cache_manager,
            )
            attn_output = torch.empty_like(query_layer.contiguous().view(-1, self.projection_size))

            # NOTE: no bug in context attn fwd (del it )
            lightllm_llama2_context_attention_fwd(
                query_layer,
                key_layer,
                value_layer,
                attn_output.view(-1, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head),
                infer_state.start_loc,
                infer_state.seq_len,
                infer_state.max_len_in_batch,
            )

        else:
            if infer_state.decode_is_contiguous:
                # if decode is contiguous, then we copy to key cache and value cache in cache manager directly
                cache_k = infer_state.cache_manager.key_buffer[infer_state.decode_layer_id][
                    infer_state.decode_mem_start : infer_state.decode_mem_end, :, :
                ]
                cache_v = infer_state.cache_manager.value_buffer[infer_state.decode_layer_id][
                    infer_state.decode_mem_start : infer_state.decode_mem_end, :, :
                ]
                cache_k.copy_(key_layer)
                cache_v.copy_(value_layer)
            else:
                # if decode is not contiguous, use triton kernel to copy key and value cache
                # k, v shape: [batch_size, num_heads, head_dim/embed_size_per_head
                copy_kv_to_mem_cache(
                    infer_state.decode_layer_id,
                    key_layer,
                    value_layer,
                    infer_state.decode_mem_index,
                    infer_state.cache_manager,
                )

            # second token and follows
            attn_output = torch.empty_like(query_layer.contiguous().view(-1, self.projection_size))
            cache_k = infer_state.cache_manager.key_buffer[infer_state.decode_layer_id][
                : infer_state.decode_mem_end, :, :
            ]
            cache_v = infer_state.cache_manager.value_buffer[infer_state.decode_layer_id][
                : infer_state.decode_mem_end, :, :
            ]

            # ==================================
            # core attention computation is replaced by triton kernel
            # ==================================
            Llama2TokenAttentionForwards.token_attn(
                query_layer,
                cache_k,
                cache_v,
                attn_output,
                infer_state.block_loc,
                infer_state.start_loc,
                infer_state.seq_len,
                infer_state.max_len_in_batch,
                infer_state.other_kv_index,
            )

            # print('after attention',torch.isnan(attn_output).any())

        # =================
        # Output:[b,sq, h]
        # =================
        output = self.dense(attn_output).reshape(batch_size, -1, hidden_size)

        return output, kv_cache

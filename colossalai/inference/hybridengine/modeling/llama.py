# This code is adapted from huggingface transformers: https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/llama/modeling_llama.py
import math
from typing import List, Optional, Tuple

import torch
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM, LlamaModel
from transformers.utils import logging

from colossalai.inference.tensor_parallel.batch_infer_state import BatchInferState
from colossalai.kernel.triton import llama_context_attn_fwd, token_attention_fwd
from colossalai.kernel.triton.token_attention_kernel import Llama2TokenAttentionForwards
from colossalai.pipeline.stage_manager import PipelineStageManager

from ._utils import copy_kv_to_mem_cache

try:
    from lightllm.models.llama2.triton_kernel.context_flashattention_nopad import (
        context_attention_fwd as lightllm_llama2_context_attention_fwd,
    )
    from lightllm.models.llama.triton_kernel.context_flashattention_nopad import (
        context_attention_fwd as lightllm_context_attention_fwd,
    )
    from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd as llama_rotary_embedding_fwd

    HAS_LIGHTLLM_KERNEL = True
except:
    print("please install lightllm from source to run inference: https://github.com/ModelTC/lightllm")
    HAS_LIGHTLLM_KERNEL = False

try:
    from flash_attn import flash_attn_with_kvcache

    HAS_FLASH_KERNEL = True
except:
    HAS_FLASH_KERNEL = False
    print("please install flash attentiom from https://github.com/Dao-AILab/flash-attention")


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def llama_triton_context_attention(
    query_states, key_states, value_states, attn_output, infer_state, num_key_value_groups=1
):
    if num_key_value_groups == 1:
        if HAS_LIGHTLLM_KERNEL is False:
            llama_context_attn_fwd(
                query_states,
                key_states,
                value_states,
                attn_output,
                infer_state.start_loc,
                infer_state.seq_len,
                # infer_state.cache_manager.past_key_values_length,
                infer_state.max_len_in_batch,
            )
        else:
            lightllm_context_attention_fwd(
                query_states,
                key_states,
                value_states,
                attn_output,
                infer_state.start_loc,
                infer_state.seq_len,
                # infer_state.cache_manager.past_key_values_length,
                infer_state.max_len_in_batch,
            )
    else:
        assert HAS_LIGHTLLM_KERNEL is True, "You have to install lightllm kernels to run llama2 model"
        lightllm_llama2_context_attention_fwd(
            query_states,
            key_states,
            value_states,
            attn_output,
            infer_state.start_loc,
            infer_state.seq_len,
            # infer_state.cache_manager.past_key_values_length,
            infer_state.max_len_in_batch,
        )


def llama_triton_token_attention(query_states, attn_output, infer_state, num_key_value_groups=1):
    assert HAS_LIGHTLLM_KERNEL is True, "You have to install lightllm kernel to run token attention for llama models"
    if num_key_value_groups == 1:
        token_attention_fwd(
            query_states,
            infer_state.cache_manager.key_buffer[infer_state.decode_layer_id],
            infer_state.cache_manager.value_buffer[infer_state.decode_layer_id],
            attn_output,
            infer_state.block_loc,
            infer_state.start_loc,
            infer_state.seq_len,
            # infer_state.cache_manager.past_key_values_length,
            infer_state.max_len_in_batch,
        )
    else:
        Llama2TokenAttentionForwards.token_attn(
            query_states,
            infer_state.cache_manager.key_buffer[infer_state.decode_layer_id],
            infer_state.cache_manager.value_buffer[infer_state.decode_layer_id],
            attn_output,
            infer_state.block_loc,
            infer_state.start_loc,
            infer_state.seq_len,
            # infer_state.cache_manager.past_key_values_length,
            infer_state.max_len_in_batch,
            infer_state.other_kv_index,
        )


class LlamaInferenceForwards:
    """
    This class holds forwards for llama inference.
    We intend to replace the forward methods for LlamaModel, LlamaDecoderLayer, and LlamaAttention for LlamaForCausalLM.
    """

    @staticmethod
    def llama_causal_lm_forward(
        self: LlamaForCausalLM,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        infer_state: BatchInferState = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        """
        logger = logging.get_logger(__name__)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False

        # If is first stage and after warmup, go throught lm_head first
        if stage_manager.is_first_stage() and hidden_states is not None:
            lm_logits = self.lm_head(hidden_states)
            return {"logits": lm_logits}

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = LlamaInferenceForwards.llama_model_forward(
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            infer_state=infer_state,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
        )

        return outputs

    @staticmethod
    def llama_model_forward(
        self: LlamaModel,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        infer_state: BatchInferState = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # retrieve input_ids and inputs_embeds
        if stage_manager is None or stage_manager.is_first_stage():
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
            elif input_ids is not None:
                batch_size, seq_length = input_ids.shape
            elif inputs_embeds is not None:
                batch_size, seq_length, _ = inputs_embeds.shape
            else:
                raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            hidden_states = inputs_embeds
        else:
            assert stage_manager is not None
            assert hidden_states is not None, f"hidden_state should not be none in stage {stage_manager.stage}"
            input_shape = hidden_states.shape[:-1]
            batch_size, seq_length = input_shape
            device = hidden_states.device

        if infer_state.is_context_stage:
            past_key_values_length = 0
        else:
            past_key_values_length = infer_state.max_len_in_batch - 1

        # NOTE: differentiate with prefill stage
        #       block_loc require different value-assigning method for two different stage
        if use_cache and seq_length != 1:
            # NOTE assume prefill stage
            # allocate memory block
            infer_state.is_context_stage = True  # set prefill stage, notify attention layer
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
                infer_state.block_loc[:, infer_state.max_len_in_batch - 1] = infer_state.decode_mem_index
            else:
                infer_state.decode_is_contiguous = False
                alloc_mem = infer_state.cache_manager.alloc(batch_size)
                infer_state.decode_mem_index = alloc_mem
                infer_state.block_loc[:, infer_state.max_len_in_batch - 1] = infer_state.decode_mem_index

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.repeat(batch_size, 1)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

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

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, infer_state.max_len_in_batch), dtype=torch.bool, device=hidden_states.device
            )

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        # decoder layers
        infer_state.decode_layer_id = 0

        start_idx, end_idx = stage_index[0], stage_index[1]
        if past_key_values is None:
            past_key_values = tuple([None] * (end_idx - start_idx + 1))

        for idx, past_key_value in zip(range(start_idx, end_idx), past_key_values):
            decoder_layer = self.layers[idx]
            # NOTE: modify here for passing args to decoder layer
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                infer_state=infer_state,
            )
            infer_state.decode_layer_id += 1
            hidden_states = layer_outputs[0]

        if stage_manager.is_last_stage() or stage_manager.num_stages == 1:
            hidden_states = self.norm(hidden_states)

        # update indices
        # infer_state.block_loc[:, infer_state.max_len_in_batch-1] = infer_state.total_token_num + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        infer_state.start_loc += torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        infer_state.seq_len += 1
        infer_state.max_len_in_batch += 1

        # if not return_dict:
        #     return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        # return BaseModelOutputWithPast(
        #     last_hidden_state=hidden_states,
        #     past_key_values=next_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attns,
        # )
        return {"hidden_states": hidden_states}

    @staticmethod
    def llama_decoder_layer_forward(
        self: LlamaDecoderLayer,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        infer_state: Optional[BatchInferState] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            infer_state=infer_state,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    @staticmethod
    def llama_flash_attn_kvcache_forward(
        self: LlamaAttention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        infer_state: Optional[BatchInferState] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        assert use_cache is True, "use_cache should be set to True using this llama attention"

        bsz, q_len, _ = hidden_states.size()

        # NOTE might think about better way to handle transposed k and v
        # key_states            [bs, seq_len, num_heads, head_dim/embed_size_per_head]
        # key_states_transposed [bs, num_heads, seq_len, head_dim/embed_size_per_head]

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # NOTE might want to revise
        #   need some way to record the length of past key values cache
        #   since we won't return past_key_value_cache right now

        cos, sin = infer_state.position_cos, infer_state.position_sin

        llama_rotary_embedding_fwd(query_states.view(-1, self.num_heads, self.head_dim), cos, sin)
        llama_rotary_embedding_fwd(key_states.view(-1, self.num_key_value_heads, self.head_dim), cos, sin)

        query_states = query_states.reshape(-1, self.num_heads, self.head_dim)
        key_states = key_states.reshape(-1, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(-1, self.num_key_value_heads, self.head_dim)

        if infer_state.is_context_stage:
            # first token generation
            # copy key and value calculated in current step to memory manager
            copy_kv_to_mem_cache(
                infer_state.decode_layer_id,
                key_states,
                value_states,
                infer_state.context_mem_index,
                infer_state.cache_manager,
            )
            attn_output = torch.empty_like(query_states)

            llama_triton_context_attention(
                query_states,
                key_states,
                value_states,
                attn_output,
                infer_state,
                num_key_value_groups=self.num_key_value_groups,
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
                cache_k.copy_(key_states)
                cache_v.copy_(value_states)
            else:
                # if decode is not contiguous, use triton kernel to copy key and value cache
                # k, v shape: [batch_size, num_heads, head_dim/embed_size_per_head
                copy_kv_to_mem_cache(
                    infer_state.decode_layer_id,
                    key_states,
                    value_states,
                    infer_state.decode_mem_index,
                    infer_state.cache_manager,
                )

            if HAS_LIGHTLLM_KERNEL:
                attn_output = torch.empty_like(query_states)
                llama_triton_token_attention(
                    query_states, attn_output, infer_state, num_key_value_groups=self.num_key_value_groups
                )
            else:
                self.num_heads // self.num_key_value_heads
                cache_k = infer_state.cache_manager.key_buffer[infer_state.decode_layer_id]
                cache_v = infer_state.cache_manager.value_buffer[infer_state.decode_layer_id]

                query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim)
                copy_cache_k = cache_k.view(bsz, -1, self.num_key_value_heads, self.head_dim)
                copy_cache_v = cache_v.view(bsz, -1, self.num_key_value_heads, self.head_dim)

                attn_output = flash_attn_with_kvcache(
                    q=query_states,
                    k_cache=copy_cache_k,
                    v_cache=copy_cache_v,
                    softmax_scale=1 / math.sqrt(self.head_dim),
                    causal=True,
                )

        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        # return past_key_value as None
        return attn_output, None, None

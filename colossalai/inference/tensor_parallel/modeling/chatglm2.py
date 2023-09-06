from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaModel, LlamaRMSNorm

from colossalai.inference.tensor_parallel.batch_infer_state import BatchInferState
from colossalai.kernel.triton.context_attention import llama_context_attn_fwd
from colossalai.kernel.triton.copy_kv_cache_dest import copy_kv_cache_to_dest
from colossalai.kernel.triton.rotary_embedding_kernel import rotary_embedding_fwd
from colossalai.kernel.triton.token_attention_kernel import token_attention_fwd
from colossalai.shardformer.modeling.chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMModel

try:
    from vllm import layernorm_ops, pos_encoding_ops
    rms_norm = layernorm_ops.rms_norm
    rotary_embedding_neox = pos_encoding_ops.rotary_embedding_neox
    HAS_VLLM_KERNERL = True
except:
    print("fall back to original rotary_embedding_neox of huggingface")
    print("install vllm from https://github.com/vllm-project/vllm to accelerate your inference")
    print(
        "if falied to install vllm, please use this branch to install: https://github.com/tiandiao123/vllm/tree/setup_branch"
    )
    HAS_VLLM_KERNERL = False


class ChatGLM2InferenceForwards:
    """
    This class holds forwards for Chatglm2 inference.
    We intend to replace the forward methods for ChatGLMModel, ChatGLMEecoderLayer, and ChatGLMAttention.
    """

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
    ):
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (return_dict if return_dict is not None else self.config.use_return_dict)

        batch_size, seq_length = input_ids.shape

        infer_state = self.infer_state
        seq_length_with_past = seq_length
        past_key_values_length = 0

        # prefill stage at first
        if use_cache and seq_length != 1:
            infer_state.is_context_stage = True
            infer_state.context_mem_index = infer_state.cache_manager.alloc(infer_state.total_token_num)
            infer_state.init_block_loc(infer_state.block_loc, infer_state.seq_len, seq_length,
                                       infer_state.context_mem_index)
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

        if infer_state.is_context_stage:

            infer_state.position_cos = torch.index_select(self._cos_cached, 0, position_ids.view(-1)).view(
                position_ids.view(-1).shape[0], -1)
            infer_state.position_sin = torch.index_select(self._sin_cached, 0, position_ids.view(-1)).view(
                position_ids.view(-1).shape[0], -1)
        else:
            seq_len = infer_state.seq_len
            infer_state.position_cos = torch.index_select(self._cos_cached, 0, seq_len - 1).view(seq_len.shape[0], -1)
            infer_state.position_sin = torch.index_select(self._sin_cached, 0, seq_len - 1).view(seq_len.shape[0], -1)

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
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds,
            full_attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

        if not return_dict:
            return tuple(v for v in [
                hidden_states,
                presents,
                all_hidden_states,
                all_self_attentions,
            ] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

from typing import List, Optional, Tuple

import torch
import numpy as np
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from transformers.models.llama.modeling_llama import LlamaModel, LlamaRMSNorm, LlamaAttention
from colossalai.shardformer.inference import BatchInferState
from colossalai.kernel.triton.copy_kv_cache_dest import copy_kv_cache_to_dest
from colossalai.kernel.triton.context_attention import llama_context_attn_fwd
from colossalai.kernel.triton.token_attention_kernel import token_attention_fwd

class LlamaInferenceForwards:
    """
    This class holds forwards for llama inference.
    """

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
    ):
        batch_size = input_ids.shape[0] # input_ids.shape[0]

        infer_info = BatchInferState(batch_size, input_ids.shape[1])
        infer_info.batch_size = batch_size
        # NOTE: dummy implementation here for testing, just assume all inputs same length
        infer_info.block_loc = self.block_loc
        infer_info.start_loc = self.start_loc
        infer_info.seq_len = self.seq_len
        infer_info.max_len_in_batch = self.max_len_in_batch

        b_seq_len_numpy = infer_info.seq_len.cpu().numpy()
        position_ids = torch.from_numpy(np.concatenate([np.arange(0, b_seq_len_numpy[i])
                                                for i in range(len(b_seq_len_numpy))], axis=0)).cuda()
        
        # this equals
        infer_info.position_cos = torch.index_select(self._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
        infer_info.position_sin = torch.index_select(self._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            # TODO dummy but work, revise it
            past_key_values_length = self.cache_manager.past_key_values_length
            # past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        infer_info.set_cache_manager(self.cache_manager)
        
        # FIXME: differentiate with prefill stage
        #       block_loc require different value-assigning method for two different stage
        if use_cache and seq_length != 1:
            # NOTE assuem prefill stage
            # allocate memory block
            infer_info.is_context_stage = True  # set prefill stage, notify attention layer
            infer_info.context_mem_index = infer_info.cache_manager.alloc(infer_info.total_token_num)
            infer_info.init_block_loc(infer_info.block_loc, infer_info.seq_len, seq_length, infer_info.context_mem_index)
        else:
            # TODO handle the condition that no contiguous memory presents 
            alloc_mem = self.cache_manager.alloc_contiguous(batch_size)
            if alloc_mem is not None:
                infer_info.decode_mem_index = alloc_mem[0]
                infer_info.decode_mem_start = alloc_mem[1]
                infer_info.decode_mem_end = alloc_mem[2]
                infer_info.block_loc[:, seq_length_with_past - 1] = infer_info.decode_mem_index
            else:
                print(f" *** Encountered allocation non-contiguous")
                print(f"    infer_info.cache_manager.past_key_values_length: {infer_info.cache_manager.past_key_values_length}")
                infer_info.decode_is_contiguous = False
                alloc_mem = self.cache_manager.alloc(batch_size)
                infer_info.decode_mem_index = alloc_mem
                # infer_info.decode_key_buffer = torch.empty((batch_size, self.tp_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
                # infer_info.decode_value_buffer = torch.empty((batch_size, self.tp_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
                infer_info.block_loc[:, seq_length_with_past - 1] = infer_info.decode_mem_index

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        infer_info.decode_layer_id = 0

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            # NOTE: modify here for passing args to decoder layer
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                infer_info=infer_info,
            )
            infer_info.decode_layer_id += 1
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        hidden_states = self.norm(hidden_states)
        next_cache = next_decoder_cache if use_cache else None

        # update indices
        self.max_len_in_batch += 1
        self.block_loc[:, self.max_len_in_batch-1] = self.total_token_num + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        self.start_loc = self.start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        self.total_token_num += batch_size
        self.seq_len += 1

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        
    @staticmethod
    def llama_decoder_layer_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        infer_info: Optional[BatchInferState] = None,
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
            infer_info=infer_info,
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
        infer_info: Optional[BatchInferState] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        assert use_cache is True, "use_cache should be set to True using this llama attention"

        bsz, q_len, _ = hidden_states.size()
            
        # TODO might think about better way to handle transposed k and v
        # key_states            [bs, seq_len, num_heads, head_dim/embed_size_per_head]
        # key_states_transposed [bs, num_heads, seq_len, head_dim/embed_size_per_head]

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states_transposed = key_states.transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        
        # cos, sin = self.rotary_emb(value_states_transposed, seq_len=kv_seq_len)
        cos ,sin = infer_info.position_cos, infer_info.position_sin
    
        cos_sin_cache = torch.cat((cos, sin), dim=-1)
        
        from col_pos_encoding_ops import rotary_embedding_neox
        
        rotary_embedding_neox(position_ids, query_states, key_states_transposed, self.head_dim, cos_sin_cache)
        
        def _copy_kv_to_mem_cache(layer_id, key_buffer, value_buffer, context_mem_index, mem_manager):
            num_heads = key_buffer.shape[2]
            head_dim = key_buffer.shape[3]
            key_buffer = key_buffer.view(-1, num_heads, head_dim)
            value_buffer = value_buffer.view(-1, num_heads, head_dim)
            copy_kv_cache_to_dest(key_buffer, context_mem_index, mem_manager.key_buffer[layer_id])
            copy_kv_cache_to_dest(value_buffer, context_mem_index, mem_manager.value_buffer[layer_id])
            return

        # copy key and value calculated in current step to memory manager
        if infer_info.is_context_stage:
            _copy_kv_to_mem_cache(infer_info.decode_layer_id, key_states, value_states, infer_info.context_mem_index, infer_info.cache_manager)
        else:
            _copy_kv_to_mem_cache(infer_info.decode_layer_id, key_states, value_states, infer_info.decode_mem_index, infer_info.cache_manager)

        # this is worse than destcopy
        # torch.Tensor.copy_(infer_info.cache_manager.key_buffer[infer_info.decode_layer_id][infer_info.decode_mem_start:infer_info.decode_mem_end, :, :],key_states)
        # torch.Tensor.copy_(infer_info.cache_manager.value_buffer[infer_info.decode_layer_id][infer_info.decode_mem_start:infer_info.decode_mem_end, :, :],value_states)

        # FIXME might want to revise
        #   need some way to record the length of past key values cache
        #   since we won't return past_key_value_cache right now
        if infer_info.decode_layer_id == 0:  # once per model.forward
            infer_info.cache_manager.past_key_values_length += q_len  # seq_len

        query_states = query_states.transpose(1, 2)

        if infer_info.is_context_stage:
            # first token generation

            # attn_output, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(query_states, 
            #                                 key_states,
            #                                 value_states, 
            #                                 0,
            #                                 1/math.sqrt(self.head_dim), 
            #                                 causal,
            #                                 False)

            attn_output = torch.empty_like(query_states)

            # calcu_shape for context_attention_fwd
            calcu_shape1 = (-1, self.num_heads, self.head_dim)

            llama_context_attn_fwd(query_states.view(calcu_shape1),
                                key_states.view(calcu_shape1),
                                value_states.view(calcu_shape1),
                                attn_output.view(calcu_shape1),
                                infer_info.start_loc,
                                infer_info.seq_len,
                                infer_info.max_len_in_batch)
            
        else:
            # second token and follows
            # kv = torch.stack((key_states, value_states), dim=2)
            # (batch_size, seqlen, nheads, headdim)
            attn_output = torch.empty_like(query_states)
            
            token_attention_fwd(query_states,
                                infer_info.cache_manager.key_buffer[infer_info.decode_layer_id],
                                infer_info.cache_manager.value_buffer[infer_info.decode_layer_id],
                                attn_output,
                                infer_info.block_loc,
                                infer_info.start_loc,
                                infer_info.seq_len,
                                infer_info.max_len_in_batch)

        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        # return past_key_value as None 
        return attn_output, None, None
    
    
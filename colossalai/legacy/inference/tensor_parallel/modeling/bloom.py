import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers.models.bloom.modeling_bloom import (
    BaseModelOutputWithPastAndCrossAttentions,
    BloomAttention,
    BloomBlock,
    BloomForCausalLM,
    BloomModel,
    CausalLMOutputWithCrossAttentions,
)
from transformers.utils import logging

from colossalai.inference.tensor_parallel.batch_infer_state import BatchInferState
from colossalai.kernel.triton import bloom_context_attn_fwd, copy_kv_cache_to_dest, token_attention_fwd

try:
    from lightllm.models.bloom.triton_kernel.context_flashattention_nopad import (
        context_attention_fwd as lightllm_bloom_context_attention_fwd,
    )

    HAS_LIGHTLLM_KERNEL = True
except:
    HAS_LIGHTLLM_KERNEL = False


def generate_alibi(n_head, dtype=torch.float16):
    """
    This method is adapted from `_generate_alibi` function
    in `lightllm/models/bloom/layer_weights/transformer_layer_weight.py`
    of the ModelTC/lightllm GitHub repository.
    This method is originally the `build_alibi_tensor` function
    in `transformers/models/bloom/modeling_bloom.py`
    of the huggingface/transformers GitHub repository.
    """

    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start**i for i in range(n)]

    def get_slopes(n):
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            slopes_power_of_2 = get_slopes_power_of_2(closest_power_of_2)
            slopes_double = get_slopes(2 * closest_power_of_2)
            slopes_combined = slopes_power_of_2 + slopes_double[0::2][: n - closest_power_of_2]
            return slopes_combined

    slopes = get_slopes(n_head)
    return torch.tensor(slopes, dtype=dtype)


class BloomInferenceForwards:
    """
    This class serves a micro library for bloom inference forwards.
    We intend to replace the forward methods for BloomForCausalLM, BloomModel, BloomBlock, and BloomAttention,
    as well as prepare_inputs_for_generation method for BloomForCausalLM.
    For future improvement, we might want to skip replacing methods for BloomForCausalLM,
    and call BloomModel.forward iteratively in TpInferEngine
    """

    @staticmethod
    def bloom_model_forward(
        self: BloomModel,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        infer_state: Optional[BatchInferState] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        logger = logging.get_logger(__name__)

        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # still need to keep past_key_values to fit original forward flow
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # NOTE determine if BatchInferState is passed in via arg
        #      if not, get the attr binded to the model
        # We might wantto remove setattr later
        if infer_state is None:
            assert hasattr(self, "infer_state")
            infer_state = self.infer_state

        # infer_state.cache_manager = self.cache_manager
        if infer_state.is_context_stage:
            past_key_values_length = 0
        else:
            past_key_values_length = infer_state.max_len_in_batch - 1

        if use_cache and seq_length != 1:
            # prefill stage
            infer_state.is_context_stage = True  # set prefill stage, notify attention layer
            infer_state.context_mem_index = infer_state.cache_manager.alloc(infer_state.total_token_num)
            BatchInferState.init_block_loc(
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
                print(f" *** Encountered allocation non-contiguous")
                print(f"    infer_state.max_len_in_batch : {infer_state.max_len_in_batch}")
                infer_state.decode_is_contiguous = False
                alloc_mem = infer_state.cache_manager.alloc(batch_size)
                infer_state.decode_mem_index = alloc_mem
                # infer_state.decode_key_buffer = torch.empty((batch_size, self.tp_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
                # infer_state.decode_value_buffer = torch.empty((batch_size, self.tp_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
                infer_state.block_loc[:, infer_state.max_len_in_batch - 1] = infer_state.decode_mem_index

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, infer_state.max_len_in_batch), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        # NOTE revise: we might want to store a single 1D alibi(length is #heads) in model,
        #      or store to BatchInferState to prevent re-calculating
        #      When we have multiple process group (e.g. dp together with tp), we need to pass the pg to here
        # alibi = generate_alibi(self.num_heads).contiguous().cuda()
        tp_size = dist.get_world_size()
        curr_tp_rank = dist.get_rank()
        alibi = (
            generate_alibi(self.num_heads * tp_size)
            .contiguous()[curr_tp_rank * self.num_heads : (curr_tp_rank + 1) * self.num_heads]
            .cuda()
        )
        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        infer_state.decode_layer_id = 0
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # NOTE: currently our KV cache manager does not handle this condition
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache=use_cache, output_attentions=output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    alibi,
                    causal_mask,
                    layer_past,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                    infer_state=infer_state,
                )

            infer_state.decode_layer_id += 1
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # update indices of kv cache block
        # NOT READY FOR PRIME TIME
        # might want to remove this part, instead, better to pass the BatchInferState from model forward,
        #       and update these information in engine.generate after model foward called
        infer_state.start_loc = infer_state.start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        infer_state.seq_len += 1
        infer_state.max_len_in_batch += 1

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,  # should always be (None, None, ..., None)
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    @staticmethod
    def bloom_for_causal_lm_forward(
        self: BloomForCausalLM,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        infer_state: Optional[BatchInferState] = None,
        **deprecated_arguments,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        logging.get_logger(__name__)

        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = BloomInferenceForwards.bloom_model_forward(
            self.transformer,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            infer_state=infer_state,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def bloom_for_causal_lm_prepare_inputs_for_generation(
        self: BloomForCausalLM,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # only last token for input_ids if past is not None
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

            # NOTE we won't use past key values here
            # the cache may be in the stardard format (e.g. in contrastive search), convert to bloom's format if needed
            # if past_key_values[0][0].shape[0] == input_ids.shape[0]:
            #     past_key_values = self._convert_to_bloom_cache(past_key_values)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def bloom_block_forward(
        self: BloomBlock,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        infer_state: Optional[BatchInferState] = None,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            infer_state=infer_state,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        layernorm_output = self.post_attention_layernorm(attention_output)

        # Get residual
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP.
        output = self.mlp(layernorm_output, residual)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions

    @staticmethod
    def bloom_attention_forward(
        self: BloomAttention,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        infer_state: Optional[BatchInferState] = None,
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
        batch_size, q_length, H, D_HEAD = query_layer.shape
        k = key_layer.reshape(-1, H, D_HEAD)  # batch_size * q_length, H, D_HEAD, q_lenth == 1
        v = value_layer.reshape(-1, H, D_HEAD)  # batch_size * q_length, H, D_HEAD, q_lenth == 1

        mem_manager = infer_state.cache_manager
        layer_id = infer_state.decode_layer_id

        if infer_state.is_context_stage:
            # context process
            max_input_len = q_length
            b_start_loc = infer_state.start_loc
            b_seq_len = infer_state.seq_len[:batch_size]
            q = query_layer.reshape(-1, H, D_HEAD)

            copy_kv_cache_to_dest(k, infer_state.context_mem_index, mem_manager.key_buffer[layer_id])
            copy_kv_cache_to_dest(v, infer_state.context_mem_index, mem_manager.value_buffer[layer_id])

            # output = self.output[:batch_size*q_length, :, :]
            output = torch.empty_like(q)

            if HAS_LIGHTLLM_KERNEL:
                lightllm_bloom_context_attention_fwd(q, k, v, output, alibi, b_start_loc, b_seq_len, max_input_len)
            else:
                bloom_context_attn_fwd(q, k, v, output, b_start_loc, b_seq_len, max_input_len, alibi)

            context_layer = output.view(batch_size, q_length, H * D_HEAD)
        else:
            # query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
            # need shape: batch_size, H, D_HEAD (q_length == 1), input q shape : (batch_size, q_length(1), H, D_HEAD)
            assert q_length == 1, "for non-context process, we only support q_length == 1"
            q = query_layer.reshape(-1, H, D_HEAD)

            if infer_state.decode_is_contiguous:
                # if decode is contiguous, then we copy to key cache and value cache in cache manager directly
                cache_k = infer_state.cache_manager.key_buffer[layer_id][
                    infer_state.decode_mem_start : infer_state.decode_mem_end, :, :
                ]
                cache_v = infer_state.cache_manager.value_buffer[layer_id][
                    infer_state.decode_mem_start : infer_state.decode_mem_end, :, :
                ]
                cache_k.copy_(k)
                cache_v.copy_(v)
            else:
                # if decode is not contiguous, use triton kernel to copy key and value cache
                # k, v shape: [batch_size, num_heads, head_dim/embed_size_per_head]
                copy_kv_cache_to_dest(k, infer_state.decode_mem_index, mem_manager.key_buffer[layer_id])
                copy_kv_cache_to_dest(v, infer_state.decode_mem_index, mem_manager.value_buffer[layer_id])

            b_start_loc = infer_state.start_loc
            b_loc = infer_state.block_loc
            b_seq_len = infer_state.seq_len
            output = torch.empty_like(q)
            token_attention_fwd(
                q,
                mem_manager.key_buffer[layer_id],
                mem_manager.value_buffer[layer_id],
                output,
                b_loc,
                b_start_loc,
                b_seq_len,
                infer_state.max_len_in_batch,
                alibi,
            )

            context_layer = output.view(batch_size, q_length, H * D_HEAD)

        # NOTE: always set present as none for now, instead of returning past key value to the next decoding,
        #       we create the past key value pair from the cache manager
        present = None

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

        # dropout is not required here during inference
        output_tensor = residual + output_tensor

        outputs = (output_tensor, present)
        assert output_attentions is False, "we do not support output_attentions at this time"

        return outputs

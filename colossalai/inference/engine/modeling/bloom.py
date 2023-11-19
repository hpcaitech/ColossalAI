import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.nn import functional as F
from transformers.models.bloom.modeling_bloom import (
    BaseModelOutputWithPastAndCrossAttentions,
    BloomAttention,
    BloomBlock,
    BloomForCausalLM,
    BloomModel,
)
from transformers.utils import logging

from colossalai.inference.kv_cache.batch_infer_state import BatchInferState
from colossalai.kernel.triton import bloom_context_attn_fwd, copy_kv_cache_to_dest, token_attention_fwd
from colossalai.pipeline.stage_manager import PipelineStageManager

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
    def bloom_for_causal_lm_forward(
        self: BloomForCausalLM,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = False,
        infer_state: BatchInferState = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        tp_group: Optional[dist.ProcessGroup] = None,
        **deprecated_arguments,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        logger = logging.get_logger(__name__)

        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )

        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False

        # If is first stage and hidden_states is not None, go throught lm_head first
        if stage_manager.is_first_stage() and hidden_states is not None:
            lm_logits = self.lm_head(hidden_states)
            return {"logits": lm_logits}

        outputs = BloomInferenceForwards.bloom_model_forward(
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
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
            tp_group=tp_group,
        )

        return outputs

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
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        infer_state: BatchInferState = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        tp_group: Optional[dist.ProcessGroup] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        logger = logging.get_logger(__name__)

        # add warnings here
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False
        if use_cache:
            logger.warning_once("use_cache=True is not supported for pipeline models at the moment.")
            use_cache = False

        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        # first stage
        if stage_manager.is_first_stage():
            # check inputs and inputs embeds
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                batch_size, seq_length = input_ids.shape
            elif inputs_embeds is not None:
                batch_size, seq_length, _ = inputs_embeds.shape
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)

            hidden_states = self.word_embeddings_layernorm(inputs_embeds)
        # other stage
        else:
            input_shape = hidden_states.shape[:-1]
            batch_size, seq_length = input_shape

        if infer_state.is_context_stage:
            past_key_values_length = 0
        else:
            past_key_values_length = infer_state.max_len_in_batch - 1

        if seq_length != 1:
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
                infer_state.block_loc[:, infer_state.max_len_in_batch - 1] = infer_state.decode_mem_index

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, infer_state.max_len_in_batch), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        # NOTE revise: we might want to store a single 1D alibi(length is #heads) in model,
        #      or store to BatchInferState to prevent re-calculating
        #      When we have multiple process group (e.g. dp together with tp), we need to pass the pg to here
        tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        curr_tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
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

        start_idx, end_idx = stage_index[0], stage_index[1]
        if past_key_values is None:
            past_key_values = tuple([None] * (end_idx - start_idx + 1))

        for idx, past_key_value in zip(range(start_idx, end_idx), past_key_values):
            block = self.h[idx]
            outputs = block(
                hidden_states,
                layer_past=past_key_value,
                attention_mask=causal_mask,
                head_mask=head_mask[idx],
                use_cache=use_cache,
                output_attentions=output_attentions,
                alibi=alibi,
                infer_state=infer_state,
            )

            infer_state.decode_layer_id += 1
            hidden_states = outputs[0]

        if stage_manager.is_last_stage() or stage_manager.num_stages == 1:
            hidden_states = self.ln_f(hidden_states)

        # update indices
        infer_state.start_loc = infer_state.start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        infer_state.seq_len += 1
        infer_state.max_len_in_batch += 1

        # always return dict for imediate stage
        return {"hidden_states": hidden_states}

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

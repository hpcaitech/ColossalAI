from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2DoubleHeadsModel,
    GPT2DoubleHeadsModelOutput,
    GPT2ForQuestionAnswering,
    GPT2ForSequenceClassification,
    GPT2ForTokenClassification,
    GPT2LMHeadModel,
    GPT2Model,
)
from transformers.utils import logging

from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.layer import ColoAttention, RingAttention
from colossalai.shardformer.layer._operation import gather_sp_output, split_forward_gather_backward
from colossalai.shardformer.layer.utils import is_share_sp_tp, split_batch_zigzag
from colossalai.shardformer.shard import ShardConfig

from ..layer import dist_cross_entropy

logger = logging.get_logger(__name__)


def _get_attention_mask(
    self: GPT2Model,
    shard_config: ShardConfig,
    hidden_states: torch.Tensor,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]],
    attention_mask: Optional[torch.FloatTensor],
    encoder_hidden_states: Optional[torch.Tensor],
    encoder_attention_mask: Optional[torch.FloatTensor],
) -> Tuple[Optional[Union[torch.Tensor, dict]], Optional[Union[torch.Tensor, dict]]]:
    # Received input is already split for non-first pipeline stages,
    # but attn mask isn't
    batch_size = hidden_states.size(0)
    seq_len = attention_mask.size(-1)

    sp_mode = shard_config.sequence_parallelism_mode
    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.config.add_cross_attention and encoder_hidden_states is not None:
        assert not sp_mode == "ring_attn", "Ring Attention only supports decoder-only."
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        if shard_config.enable_flash_attention:
            encoder_attention_mask = ColoAttention.prepare_attn_kwargs(
                (encoder_batch_size, 1, seq_len, encoder_sequence_length),
                dtype=hidden_states.dtype,
                dtype2=encoder_hidden_states.dtype,
                q_padding_mask=attention_mask,
                kv_padding_mask=encoder_attention_mask,
            )
        else:
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=encoder_hidden_states.device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        if shard_config.enable_flash_attention:
            encoder_attention_mask = {"attention_mask": None}
        else:
            encoder_attention_mask = None

    # GPT2Attention mask.
    past_key_values_length = 0
    if past_key_values is not None and past_key_values[0] is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
    if shard_config.enable_flash_attention:
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)

        attention_mask = ColoAttention.prepare_attn_kwargs(
            (batch_size, 1, seq_len, seq_len + past_key_values_length),
            hidden_states.dtype,
            hidden_states.device,
            attention_mask,
            is_causal=True,
        )
    elif attention_mask is not None:
        if batch_size <= 0:
            raise ValueError("batch_size has to be defined and > 0")
        attention_mask = attention_mask.view(batch_size, -1)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
    return attention_mask, encoder_attention_mask


class GPT2PipelineForwards:
    """
    This class serves as a micro library for forward function substitution of GPT2 models
    under pipeline setting.
    """

    @staticmethod
    def gpt2_model_forward(
        self: GPT2Model,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
        force_sp_gather: Optional[bool] = True,
    ) -> Union[Dict, Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # This function is modified on the basis of transformers.models.gpt2.modeling_gpt2.GPT2Model.forward.
        # Please refer to original code of transformers for more details.

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        logger = logging.get_logger(__name__)

        # Preprocess passed in arguments
        # TODO(baizhou): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if past_key_values:
            logger.warning_once("Non-empty past_key_values is not supported for pipeline models at the moment.")
            past_key_values = None
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False
        if use_cache:
            logger.warning_once("use_cache=True is not supported for pipeline models at the moment.")
            use_cache = False

        disable_pp = stage_manager is None
        if disable_pp or stage_manager.is_first_stage():
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if token_type_ids is not None:
                token_type_ids = token_type_ids.view(-1, input_shape[-1])
        else:
            if hidden_states is None:
                raise ValueError("hidden_states shouldn't be None for stages other than the first stage.")
            input_shape = hidden_states.size()[:-1]
            device = hidden_states.device
            hidden_states = hidden_states.view((-1,) + hidden_states.shape[-2:])
            hidden_states.shape[0]

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if disable_pp or stage_manager.is_first_stage():
            if position_ids is None:
                position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0)

            if inputs_embeds is None:
                inputs_embeds = self.wte(input_ids)
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
            if token_type_ids is not None:
                token_type_embeds = self.wte(token_type_ids)
                hidden_states = hidden_states + token_type_embeds
            hidden_states = self.drop(hidden_states)

        attn_kwargs, encoder_attention_mask = _get_attention_mask(
            self,
            shard_config,
            hidden_states,
            past_key_values,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        # split the input tensor along sequence dimension
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len/TP_size, hidden_size]
        sp_mode = shard_config.sequence_parallelism_mode
        sp_group = shard_config.sequence_parallel_process_group
        if disable_pp or stage_manager.is_first_stage():
            # Ring Attention's special zigzag batch processing
            if sp_mode == "ring_attn":
                assert shard_config.enable_flash_attention, "Ring Attention inherently requires Flash Attention."
                if not attention_mask.bool().all():
                    hidden_states, attn_kwargs, position_ids = RingAttention.prepare_varlen_batch(
                        attention_mask, sp_group, hidden_states, position_ids
                    )
                else:
                    hidden_states, position_ids = split_batch_zigzag([hidden_states, position_ids], sp_group)
            # Other sp modes
            else:
                if sp_mode == "split_gather":
                    hidden_states = split_forward_gather_backward(
                        hidden_states,
                        dim=1,
                        process_group=shard_config.tensor_parallel_process_group,
                    )
        elif sp_mode == "ring_attn":
            # Later stages already received split hidden states
            _, attn_kwargs, _ = RingAttention.prepare_varlen_batch(attention_mask, sp_group)
        del attention_mask

        # Going through held blocks.
        if disable_pp:
            start_idx, end_idx = 0, len(self.h)
        else:
            start_idx, end_idx = stage_index[0], stage_index[1]

        for i in range(start_idx, end_idx):
            block = self.h[i]
            torch.cuda.set_device(hidden_states.device)
            # Ensure that attention_mask is always on the same device as hidden_states
            if torch.is_tensor(attn_kwargs):
                attn_kwargs = attn_kwargs.to(hidden_states.device)
            if isinstance(head_mask, torch.Tensor):
                head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attn_kwargs,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=None,
                    attention_mask=attn_kwargs,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

        # When sequence parallelism is done, gather the output tensor in forward and split it in backward
        gather_output = (not shard_config.parallel_output) or force_sp_gather or is_share_sp_tp(sp_mode)
        if disable_pp or stage_manager.is_last_stage():
            if gather_output:
                hidden_states = gather_sp_output(hidden_states, shard_config)

        # gather_sp_output could've changed seq length.
        input_shape = (*input_shape[:-1], hidden_states.size(-2))
        output_shape = input_shape + (hidden_states.size(-1),)

        if disable_pp or stage_manager.is_last_stage():
            hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if disable_pp or stage_manager.is_last_stage():
            if not return_dict:
                return tuple(
                    v
                    for v in [
                        hidden_states,
                        presents,
                        all_hidden_states,
                        all_self_attentions,
                        all_cross_attentions,
                    ]
                    if v is not None
                )

            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=presents,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=all_cross_attentions,
            )
        else:
            # always return dict for intermediate stage
            return {"hidden_states": hidden_states}

    @staticmethod
    def gpt2_lmhead_model_forward(
        self: GPT2LMHeadModel,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ) -> Union[Dict, Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`

        This function is modified on the basis of transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.forward.
        Please refer to original code of transformers for more details.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = GPT2PipelineForwards.gpt2_model_forward(
            self.transformer,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
            shard_config=shard_config,
            force_sp_gather=False,
        )

        # If not at the last stage, return hidden_states as in GPT2Model
        disable_pp = stage_manager is None
        if (not disable_pp) and (not stage_manager.is_last_stage()):
            return {"hidden_states": outputs["hidden_states"]}

        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)
        if shard_config.sequence_parallelism_mode == "ring_attn":
            # Split labels in a zigzag fashion too
            sp_group = shard_config.sequence_parallel_process_group
            if not attention_mask.bool().all():
                # [B, max_seqlen // sp_size]
                labels, _, _ = RingAttention.prepare_varlen_batch(attention_mask, sp_group, labels, is_label=True)
            else:
                labels = split_batch_zigzag(labels, sp_group, seq_dim=1, is_label=True)

        if labels is not None:
            loss = dist_cross_entropy(
                labels, lm_logits, shard_config, self.lm_head.out_features, self.transformer.dtype
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    @staticmethod
    def gpt2_double_heads_model_forward(
        self: GPT2DoubleHeadsModel,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        mc_token_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mc_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ) -> Union[Dict, Tuple, GPT2DoubleHeadsModelOutput]:
        r"""
        mc_token_ids (`torch.LongTensor` of shape `(batch_size, num_choices)`, *optional*, default to index of the last token of the input):
            Index of the classification token in each input sequence. Selected in the range `[0, input_ids.size(-1) -
            1]`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids`. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to
            `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size - 1]`
        mc_labels (`torch.LongTensor` of shape `(batch_size)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where *num_choices* is the size of the second dimension of the input tensors. (see *input_ids* above)

        This function is modified on the basis of transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModel.forward.
        Please refer to original code of transformers for more details.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = GPT2PipelineForwards.gpt2_model_forward(
            self.transformer,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
            shard_config=shard_config,
        )

        # If not at the last stage, return hidden_states as in GPT2Model
        if not stage_manager.is_last_stage():
            return {"hidden_states": outputs["hidden_states"]}

        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        mc_loss = None
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            mc_loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
        lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits, mc_logits) + outputs[1:]
            if mc_loss is not None:
                output = (mc_loss,) + output
            return ((lm_loss,) + output) if lm_loss is not None else output

        return GPT2DoubleHeadsModelOutput(
            loss=lm_loss,
            mc_loss=mc_loss,
            logits=lm_logits,
            mc_logits=mc_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def gpt2_for_question_answering_forward(
        self: GPT2ForQuestionAnswering,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ) -> Union[Dict, Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        # This function is modified on the basis of transformers.models.gpt2.modeling_gpt2.GPT2ForQuestionAnswering.forward.
        # Please refer to original code of transformers for more details.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = GPT2PipelineForwards.gpt2_model_forward(
            self.transformer,
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
            shard_config=shard_config,
        )

        # If not at the last stage, return hidden_states as in GPT2Model
        if not stage_manager.is_last_stage():
            return {"hidden_states": outputs["hidden_states"]}

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def gpt2_for_token_classification_forward(
        self: GPT2ForTokenClassification,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ) -> Union[Dict, Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        # This function is modified on the basis of transformers.models.gpt2.modeling_gpt2.GPT2ForTokenClassification.forward.
        # Please refer to original code of transformers for more details.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = GPT2PipelineForwards.gpt2_model_forward(
            self.transformer,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
            shard_config=shard_config,
        )

        # If not at the last stage, return hidden_states as in GPT2Model
        if not stage_manager.is_last_stage():
            return {"hidden_states": outputs["hidden_states"]}

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def gpt2_for_sequence_classification_forward(
        self: GPT2ForSequenceClassification,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ) -> Union[Dict, Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        # This function is modified on the basis of transformers.models.gpt2.modeling_gpt2.GPT2ForSequenceClassification.forward.
        # Please refer to original code of transformers for more details.
        """
        logger = logging.get_logger(__name__)

        if input_ids is not None:
            batch_size, _ = input_ids.shape[:2]
        else:
            batch_size, _ = hidden_states.shape[:2]
        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = GPT2PipelineForwards.gpt2_model_forward(
            self.transformer,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
            shard_config=shard_config,
        )

        # If not at the last stage, return hidden_states as in GPT2Model
        if not stage_manager.is_last_stage():
            return {"hidden_states": outputs["hidden_states"]}

        hidden_states = outputs[0]
        logits = self.score(hidden_states)

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1
                logger.warning_once(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def get_gpt2_flash_attention_forward(shard_config: Optional[ShardConfig] = None):
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

    def forward(
        self: GPT2Attention,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[dict] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        assert head_mask is None, "FlashAttention does not support head_mask"
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        scale = 1.0
        if self.scale_attn_weights:
            scale /= value.size(-1) ** 0.5
        if self.scale_attn_by_inverse_layer_idx:
            scale /= float(self.layer_idx + 1)
        dropout_p = self.attn_dropout.p if self.training else 0.0

        sp_mode = shard_config.sequence_parallelism_mode
        sp_group = shard_config.sequence_parallel_process_group
        if sp_mode == "ring_attn":
            attn_output = RingAttention.attention(
                query,
                key,
                value,
                sp_group,
                **attention_mask,
                dropout_p=dropout_p,
                scale=scale,
                inner_ring_size=shard_config.inner_ring_size,
            )
        else:
            attn_output = ColoAttention.attention(query, key, value, **attention_mask, dropout_p=dropout_p, scale=scale)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output, present, None)

        return outputs

    return forward


def get_jit_fused_gpt2_mlp_forward():
    from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

    from colossalai.kernel.jit.bias_gelu import GeLUFunction as JitGeLUFunction

    def forward(self: GPT2MLP, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states, bias = self.c_fc(hidden_states)
        hidden_states = JitGeLUFunction.apply(hidden_states, bias)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

    return forward

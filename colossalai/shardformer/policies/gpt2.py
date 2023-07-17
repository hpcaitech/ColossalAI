import logging
from functools import partial
from types import MethodType
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import colossalai.shardformer.layer as col_nn
from colossalai.pipeline.stage_manager import PipelineStageManager

from .._utils import getattr_, setattr_
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = [
    'GPT2Policy', 'GPT2ModelPolicy', 'GPT2LMHeadModelPolicy', 'GPT2DoubleHeadsModelPolicy',
    'GPT2ForTokenClassificationPolicy', 'GPT2ForSequenceClassificationPolicy'
]


class GPT2Policy(Policy):

    def config_sanity_check(self):
        pass

    def preprocess(self):
        # reshape the embedding layer
        r"""
        Reshape the Embedding layer to make the embedding dimension divisible by world_size
        """
        if self.shard_config.enable_tensor_parallelism:
            vocab_size = self.model.config.vocab_size
            world_size = self.shard_config.tensor_parallel_size
            if vocab_size % world_size != 0:
                new_vocab_size = vocab_size + world_size - vocab_size % world_size
                self.model.resize_token_embeddings(new_vocab_size)
        return self.model

    def module_policy(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Model

        policy = {}

        if self.shard_config.enable_tensor_parallelism:
            policy[GPT2Model] = ModulePolicyDescription(sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="wte",
                    target_module=col_nn.VocabParallelEmbedding1D,
                ),
                SubModuleReplacementDescription(
                    suffix="drop",
                    target_module=col_nn.DropoutForParallelInput,
                ),
            ])
            policy[GPT2Block] = ModulePolicyDescription(attribute_replacement={
                "attn.embed_dim": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "attn.split_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "attn.num_heads": self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
            },
                                                        sub_module_replacement=[
                                                            SubModuleReplacementDescription(
                                                                suffix="attn.c_attn",
                                                                target_module=col_nn.GPT2FusedLinearConv1D_Col,
                                                                kwargs={
                                                                    "n_fused": 3,
                                                                },
                                                            ),
                                                            SubModuleReplacementDescription(
                                                                suffix="attn.c_proj",
                                                                target_module=col_nn.GPT2FusedLinearConv1D_Row,
                                                            ),
                                                            SubModuleReplacementDescription(
                                                                suffix="mlp.c_fc",
                                                                target_module=col_nn.GPT2FusedLinearConv1D_Col,
                                                                kwargs={
                                                                    "n_fused": 1,
                                                                },
                                                            ),
                                                            SubModuleReplacementDescription(
                                                                suffix="mlp.c_proj",
                                                                target_module=col_nn.GPT2FusedLinearConv1D_Row,
                                                            ),
                                                            SubModuleReplacementDescription(
                                                                suffix="attn.attn_dropout",
                                                                target_module=col_nn.DropoutForParallelInput,
                                                            ),
                                                            SubModuleReplacementDescription(
                                                                suffix="attn.resid_dropout",
                                                                target_module=col_nn.DropoutForParallelInput,
                                                            ),
                                                            SubModuleReplacementDescription(
                                                                suffix="mlp.dropout",
                                                                target_module=col_nn.DropoutForParallelInput,
                                                            ),
                                                        ])

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="ln_f",
                target_module=col_nn.FusedLayerNorm,
            ),
                                                        policy=policy,
                                                        target_key=GPT2Model)

            self.append_or_create_submodule_replacement(description=[
                SubModuleReplacementDescription(
                    suffix="ln_1",
                    target_module=col_nn.FusedLayerNorm,
                ),
                SubModuleReplacementDescription(
                    suffix="ln_2",
                    target_module=col_nn.FusedLayerNorm,
                ),
                SubModuleReplacementDescription(suffix="ln_cross_attn",
                                                target_module=col_nn.FusedLayerNorm,
                                                ignore_if_not_exist=True)
            ],
                                                        policy=policy,
                                                        target_key=GPT2Block)
        return policy

    def postprocess(self):
        return self.model

    def get_held_layers(self) -> List[nn.Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None

        if self.model.__class__.__name__ == 'GPT2Model':
            module = self.model
        else:
            module = self.model.transformer
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = self.distribute_layers(len(module.h), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.wte)
            held_layers.append(module.wpe)
            held_layers.append(module.drop)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.h[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.ln_f)
        return held_layers

    def set_pipeline_forward(self, model_cls: nn.Module, new_forward: Callable, policy: Dict) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
           to customized forward method, and add this changing to policy."""
        if self.pipeline_stage_manager:
            stage_manager = self.pipeline_stage_manager
            if self.model.__class__.__name__ == 'GPT2Model':
                module = self.model
            else:
                module = self.model.transformer

            layers_per_stage = Policy.distribute_layers(len(module.h), stage_manager.num_stages)
            stage_index = Policy.get_stage_index(layers_per_stage, stage_manager.stage)
            method_replacement = {'forward': partial(new_forward, stage_manager=stage_manager, stage_index=stage_index)}
            self.append_or_create_method_replacement(description=method_replacement,
                                                     policy=policy,
                                                     target_key=model_cls)


# GPT2Model
class GPT2ModelPolicy(GPT2Policy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2Model

        policy = super().module_policy()
        self.set_pipeline_forward(model_cls=GPT2Model,
                                  new_forward=GPT2PipelineForwards.gpt2_model_forward,
                                  policy=policy)
        return policy

    def get_held_layers(self) -> List[nn.Module]:
        return super().get_held_layers()

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in GPT2Model."""
        return []


# GPT2LMHeadModel
class GPT2LMHeadModelPolicy(GPT2Policy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

        module_policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            addon_module = {
                GPT2LMHeadModel:
                    ModulePolicyDescription(sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head", target_module=col_nn.Linear1D_Col, kwargs={"gather_output": True})
                    ])
            }
            module_policy.update(addon_module)

        self.set_pipeline_forward(model_cls=GPT2LMHeadModel,
                                  new_forward=GPT2PipelineForwards.gpt2_lmhead_model_forward,
                                  policy=module_policy)
        return module_policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.lm_head)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        '''The weights of wte and lm_head are shared.'''
        module = self.model
        stage_manager = self.pipeline_stage_manager
        if stage_manager and id(module.transformer.wte.weight) == id(module.lm_head.weight):
            first_stage, last_stage = 0, stage_manager.num_stages - 1
            return [{first_stage: module.transformer.wte.weight, last_stage: module.lm_head.weight}]
        else:
            return []

    def postprocess(self):
        if self.shard_config.enable_tensor_parallelism \
                and self.pipeline_stage_manager is None:
            binding_map = {"transformer.wte.weight": "lm_head.weight"}
            for k, v in binding_map.items():
                param = getattr_(self.model, k)
                setattr_(self.model, v, param)
        return self.model


# GPT2DoubleHeadsModel
class GPT2DoubleHeadsModelPolicy(GPT2Policy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModel

        module_policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            addon_module = {
                GPT2DoubleHeadsModel:
                    ModulePolicyDescription(sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head", target_module=col_nn.Linear1D_Col, kwargs={"gather_output": True})
                    ])
            }
            module_policy.update(addon_module)

        self.set_pipeline_forward(model_cls=GPT2DoubleHeadsModel,
                                  new_forward=GPT2PipelineForwards.gpt2_double_heads_model_forward,
                                  policy=module_policy)

        return module_policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            multiple_choice_head = self.model.multiple_choice_head
            held_layers.append(self.model.lm_head)
            held_layers.append(multiple_choice_head.summary)
            held_layers.append(multiple_choice_head.activation)
            held_layers.append(multiple_choice_head.first_dropout)
            held_layers.append(multiple_choice_head.last_dropout)

        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        '''The weights of wte and lm_head are shared.'''
        module = self.model
        stage_manager = self.pipeline_stage_manager
        if stage_manager and id(module.transformer.wte.weight) == id(module.lm_head.weight):
            first_stage, last_stage = 0, stage_manager.num_stages - 1
            return [{first_stage: module.transformer.wte.weight, last_stage: module.lm_head.weight}]
        else:
            return []

    def postprocess(self):
        if self.shard_config.enable_tensor_parallelism \
                and self.pipeline_stage_manager is None:
            binding_map = {"transformer.wte.weight": "lm_head.weight"}
            for k, v in binding_map.items():
                param = getattr_(self.model, k)
                setattr_(self.model, v, param)
        return self.model


# GPT2ForTokenClassification
class GPT2ForTokenClassificationPolicy(GPT2Policy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2ForTokenClassification

        module_policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            addon_module = {
                GPT2ForTokenClassification:
                    ModulePolicyDescription(sub_module_replacement=[
                        SubModuleReplacementDescription(suffix="dropout", target_module=col_nn.DropoutForParallelInput)
                    ])
            }
            module_policy.update(addon_module)

        self.set_pipeline_forward(model_cls=GPT2ForTokenClassification,
                                  new_forward=GPT2PipelineForwards.gpt2_for_token_classification_forward,
                                  policy=module_policy)
        return module_policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.dropout)
            held_layers.append(self.model.classifier)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in GPT2ForTokenClassification."""
        return []


# GPT2ForSequenceClassification
class GPT2ForSequenceClassificationPolicy(GPT2Policy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2ForSequenceClassification

        module_policy = super().module_policy()
        self.set_pipeline_forward(model_cls=GPT2ForSequenceClassification,
                                  new_forward=GPT2PipelineForwards.gpt2_for_sequence_classification_forward,
                                  policy=module_policy)
        return module_policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.score)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in GPT2ForTokenClassification."""
        return []


class GPT2PipelineForwards:
    '''
    This class serves as a micro library for forward function substitution of GPT2 models
    under pipeline setting.
    '''

    @staticmethod
    def gpt2_model_forward(
            self: 'GPT2Model',
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
            stage_index: Optional[List[int]] = None) -> Union[Tuple, 'BaseModelOutputWithPastAndCrossAttentions']:

        # This function is modified on the basis of transformers.models.gpt2.modeling_gpt2.GPT2Model.forward.
        # Please refer to original code of transformers for more details.

        from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

        # Preprocess passed in arguments
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if stage_manager.is_first_stage():
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                batch_size, seq_length = input_ids.shape
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, seq_length)
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
                batch_size = inputs_embeds.shape[0]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if token_type_ids is not None:
                token_type_ids = token_type_ids.view(-1, seq_length)
        else:
            assert hidden_states is not None
            input_shape = hidden_states.size()[:-1]
            batch_size, seq_length = input_shape[0], input_shape[1]
            device = hidden_states.device

        # GPT2Attention mask.
        if attention_mask is not None:
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
            attention_mask = attention_mask.to(dtype=self.dtype)    # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if stage_manager.is_first_stage():
            if position_ids is not None:
                position_ids = position_ids.view(-1, seq_length)
            else:
                position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

            if inputs_embeds is None:
                inputs_embeds = self.wte(input_ids)
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
            if token_type_ids is not None:
                token_type_embeds = self.wte(token_type_ids)
                hidden_states = hidden_states + token_type_embeds
                hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        # TODO: left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if past_key_values:
            logging.warning('Non-empty past_key_values is not supported for pipeline models at the moment.')
            past_key_values = None
        if output_attentions:
            logging.warning('output_attentions=True is not supported for pipeline models at the moment.')
            output_attentions = False
        if output_hidden_states:
            logging.warning('output_hidden_states=True is not supported for pipeline models at the moment.')
            output_hidden_states = False
        if use_cache:
            logging.warning('use_cache=True is not supported for pipeline models at the moment.')
            use_cache = False

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logging.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        # Going through held blocks.
        start_idx, end_idx = stage_index[0], stage_index[1]
        for i in range(start_idx, end_idx):
            block = self.h[i]
            torch.cuda.set_device(hidden_states.device)
            # Ensure that attention_mask is always on the same device as hidden_states
            if attention_mask is not None:
                attention_mask = attention_mask.to(hidden_states.device)
            if isinstance(head_mask, torch.Tensor):
                head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=None,
                    attention_mask=attention_mask,
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

        if stage_manager.is_last_stage():
            hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if stage_manager.is_last_stage():
            if not return_dict:
                return tuple(
                    v for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                    if v is not None)

            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=presents,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=all_cross_attentions,
            )
        else:
            # always return dict for intermediate stage
            return {'hidden_states': hidden_states}

    @staticmethod
    def gpt2_lmhead_model_forward(
            self: 'GPT2LMHeadModel',
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
            stage_index: Optional[List[int]] = None) -> Union[Tuple, 'CausalLMOutputWithCrossAttentions']:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
                are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`

            This function is modified on the basis of transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.forward.
            Please refer to original code of transformers for more details.
            """

        from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = GPT2PipelineForwards.gpt2_model_forward(self.transformer,
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
                                                          stage_index=stage_index)

        # If not at the last stage, return hidden_states as in GPT2Model
        if not stage_manager.is_last_stage():
            return {'hidden_states': outputs['hidden_states']}

        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
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
            self: 'GPT2DoubleHeadsModel',
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
            stage_index: Optional[List[int]] = None) -> Union[Tuple, 'GPT2DoubleHeadsModelOutput']:
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
        from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = GPT2PipelineForwards.gpt2_model_forward(self.transformer,
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
                                                          stage_index=stage_index)

        # If not at the last stage, return hidden_states as in GPT2Model
        if not stage_manager.is_last_stage():
            return {'hidden_states': outputs['hidden_states']}

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
    def gpt2_for_token_classification_forward(
            self: 'GPT2ForTokenClassification',
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
            stage_index: Optional[List[int]] = None) -> Union[Tuple, 'TokenClassifierOutput']:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        # This function is modified on the basis of transformers.models.gpt2.modeling_gpt2.GPT2ForTokenClassification.forward.
        # Please refer to original code of transformers for more details.
        """

        from transformers.modeling_outputs import TokenClassifierOutput

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = GPT2PipelineForwards.gpt2_model_forward(self.transformer,
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
                                                          stage_index=stage_index)

        # If not at the last stage, return hidden_states as in GPT2Model
        if not stage_manager.is_last_stage():
            return {'hidden_states': outputs['hidden_states']}

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
            self: 'GPT2ForSequenceClassification',
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
            stage_index: Optional[List[int]] = None) -> Union[Tuple, 'SequenceClassifierOutputWithPast']:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        # This function is modified on the basis of transformers.models.gpt2.modeling_gpt2.GPT2ForSequenceClassification.forward.
        # Please refer to original code of transformers for more details.
       """
        from transformers.modeling_outputs import SequenceClassifierOutputWithPast

        if input_ids is not None:
            batch_size, _ = input_ids.shape[:2]
        else:
            batch_size, _ = hidden_states.shape[:2]
        assert (self.config.pad_token_id is not None
                or batch_size == 1), "Cannot handle batch sizes > 1 if no padding token is defined."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = GPT2PipelineForwards.gpt2_model_forward(self.transformer,
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
                                                          stage_index=stage_index)

        # If not at the last stage, return hidden_states as in GPT2Model
        if not stage_manager.is_last_stage():
            return {'hidden_states': outputs['hidden_states']}

        hidden_states = outputs[0]
        logits = self.score(hidden_states)

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1
                logging.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`")

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

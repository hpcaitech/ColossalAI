import warnings
from functools import partial
from types import MethodType
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bloom.modeling_bloom import BloomModel
from transformers.utils import logging

import colossalai.shardformer.layer as col_nn
from colossalai.pipeline.stage_manager import PipelineStageManager

from .._utils import getattr_, setattr_
from ..modeling.bloom import build_bloom_alibi_tensor_fn
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

logger = logging.get_logger(__name__)


class BloomPolicy(Policy):

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
        from transformers.models.bloom.modeling_bloom import BloomBlock, BloomModel

        policy = {}

        if self.shard_config.enable_tensor_parallelism:
            policy[BloomBlock] = ModulePolicyDescription(attribute_replacement={
                "self_attention.hidden_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "self_attention.split_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "self_attention.num_heads": self.model.config.n_head // self.shard_config.tensor_parallel_size,
            },
                                                         sub_module_replacement=[
                                                             SubModuleReplacementDescription(
                                                                 suffix="self_attention.query_key_value",
                                                                 target_module=col_nn.Linear1D_Col,
                                                             ),
                                                             SubModuleReplacementDescription(
                                                                 suffix="self_attention.dense",
                                                                 target_module=col_nn.Linear1D_Row,
                                                             ),
                                                             SubModuleReplacementDescription(
                                                                 suffix="self_attention.attention_dropout",
                                                                 target_module=col_nn.DropoutForParallelInput,
                                                             ),
                                                             SubModuleReplacementDescription(
                                                                 suffix="mlp.dense_h_to_4h",
                                                                 target_module=col_nn.Linear1D_Col,
                                                             ),
                                                             SubModuleReplacementDescription(
                                                                 suffix="mlp.dense_4h_to_h",
                                                                 target_module=col_nn.Linear1D_Row,
                                                             ),
                                                         ])

            policy[BloomModel] = ModulePolicyDescription(
                attribute_replacement={
                    "num_heads": self.model.config.n_head // self.shard_config.tensor_parallel_size,
                },
                method_replacement={
                    "build_alibi_tensor": build_bloom_alibi_tensor_fn(self.shard_config.tensor_parallel_process_group)
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="word_embeddings",
                        target_module=col_nn.VocabParallelEmbedding1D,
                    )
                ])

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            # handle bloom model
            self.append_or_create_submodule_replacement(description=[
                SubModuleReplacementDescription(
                    suffix="ln_f",
                    target_module=col_nn.FusedLayerNorm,
                ),
                SubModuleReplacementDescription(
                    suffix="word_embeddings_layernorm",
                    target_module=col_nn.FusedLayerNorm,
                )
            ],
                                                        policy=policy,
                                                        target_key=BloomModel)

            # handle bloom block
            self.append_or_create_submodule_replacement(description=[
                SubModuleReplacementDescription(
                    suffix="input_layernorm",
                    target_module=col_nn.FusedLayerNorm,
                ),
                SubModuleReplacementDescription(
                    suffix="post_attention_layernorm",
                    target_module=col_nn.FusedLayerNorm,
                )
            ],
                                                        policy=policy,
                                                        target_key=BloomBlock)

        return policy

    def postprocess(self):
        return self.model


class BloomModelPolicy(BloomPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()
        from transformers.models.bloom.modeling_bloom import BloomModel
        if self.pipeline_stage_manager:
            stage_manager = self.pipeline_stage_manager
            layers_per_stage = Policy.distribute_layers(len(self.model.h), stage_manager.num_stages)
            stage_index = Policy.get_stage_index(layers_per_stage, stage_manager.stage)
            policy[BloomModel] = ModulePolicyDescription(method_replacement={
                "forward":
                    partial(bloom_model_forward, stage_manager=self.pipeline_stage_manager, stage_index=stage_index)
            })
        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        module = self.model
        stage_manager = self.pipeline_stage_manager
        held_layers = []
        layers_per_stage = self.distribute_layers(len(module.h), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.word_embeddings)
            held_layers.append(module.word_embeddings_layernorm)

        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.h[start_idx:end_idx])

        if stage_manager.is_last_stage():
            held_layers.append(module.ln_f)

        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        '''no shared params in bloommodel'''
        return []


class BloomForCausalLMPolicy(BloomPolicy):

    def module_policy(self):
        from transformers.models.bloom.modeling_bloom import BloomForCausalLM
        policy = super().module_policy()

        # handle tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="lm_head", target_module=col_nn.Linear1D_Col, kwargs=dict(gather_output=True)),
                                                        policy=policy,
                                                        target_key=BloomForCausalLM)

        return policy

    def postprocess(self):
        if self.shard_config.enable_tensor_parallelism:
            binding_map = {"transformer.word_embeddings.weight": "lm_head.weight"}

            for k, v in binding_map.items():
                param = getattr_(self.model, k)
                # tie weights
                setattr_(self.model, v, param)
        return self.model


class BloomForSequenceClassificationPolicy(BloomPolicy):

    def module_policy(self):
        from transformers.models.bloom.modeling_bloom import BloomForSequenceClassification
        policy = super().module_policy()

        # handle tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="score", target_module=col_nn.Linear1D_Col, kwargs=dict(gather_output=True)),
                                                        policy=policy,
                                                        target_key=BloomForSequenceClassification)

        return policy


class BloomForTokenClassificationPolicy(BloomPolicy):

    def module_policy(self):
        from transformers.models.bloom.modeling_bloom import BloomForTokenClassification
        policy = super().module_policy()

        # handle tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(description=[
                SubModuleReplacementDescription(suffix="classifier",
                                                target_module=col_nn.Linear1D_Col,
                                                kwargs=dict(gather_output=True)),
                SubModuleReplacementDescription(
                    suffix="dropout",
                    target_module=col_nn.DropoutForReplicatedInput,
                ),
            ],
                                                        policy=policy,
                                                        target_key=BloomForTokenClassification)

        return policy


class BloomForQuestionAnsweringPolicy(BloomPolicy):
    # No head sharding as the output features is only 2
    pass


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
    stage_manager: Optional[PipelineStageManager] = None,
    hidden_states: Optional[torch.FloatTensor] = None,
    stage_index: Optional[List[int]] = None,
    **deprecated_arguments,
) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
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
    output_hidden_states = (output_hidden_states
                            if output_hidden_states is not None else self.config.output_hidden_states)
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # add warnings here
    if output_attentions:
        logger.warning_once('output_attentions=True is not supported for pipeline models at the moment.')
        output_attentions = False
    if output_hidden_states:
        logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
        output_hidden_states = False
    if use_cache:
        logger.warning_once('use_cache=True is not supported for pipeline models at the moment.')
        use_cache = False
    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape batch_size x num_heads x N x N

    # head_mask has shape n_layer x batch x num_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    # case: First stage of training
    if stage_manager.is_first_stage():
        # check input_ids and inputs_embeds
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
        # initialize in the first stage and then pass to the next stage
    else:
        input_shape = hidden_states.shape[:-1]
        batch_size, seq_length = input_shape

    # extra recording tensor should be generated in the first stage

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

    if past_key_values is None:
        past_key_values = tuple([None] * len(self.h))
    # Compute alibi tensor: check build_alibi_tensor documentation,build for every stage
    seq_length_with_past = seq_length
    past_key_values_length = 0
    if past_key_values[0] is not None:
        past_key_values_length = past_key_values[0][0].shape[2]    # source_len

        seq_length_with_past = seq_length_with_past + past_key_values_length
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
    else:
        attention_mask = attention_mask.to(hidden_states.device)

    alibi = self.build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)

    # causal_mask is constructed every stage and its input is passed through different stages
    causal_mask = self._prepare_attn_mask(
        attention_mask,
        input_shape=(batch_size, seq_length),
        past_key_values_length=past_key_values_length,
    )

    start_idx, end_idx = stage_index[0], stage_index[1]
    for i, (block, layer_past) in enumerate(zip(self.h[start_idx:end_idx], past_key_values[start_idx:end_idx])):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:

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
            )

        hidden_states = outputs[0]

        if use_cache is True:
            presents = presents + (outputs[1],)
        if output_attentions:
            all_self_attentions = all_self_attentions + \
                (outputs[2 if use_cache else 1],)

    if stage_manager.is_last_stage():
        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

    # TODO: deal with all_hidden_states, all_self_attentions, presents
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if stage_manager.is_last_stage():
        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        # attention_mask is not returned ; presents = past_key_values
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    else:
        # always return dict for imediate stage
        return {'hidden_states': hidden_states}

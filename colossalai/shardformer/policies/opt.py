import warnings
from functools import partial
from typing import Callable, Dict, List

import torch.nn as nn
from torch import Tensor, nn

from colossalai.shardformer.layer import (
    FusedLayerNorm,
    LayerNorm,
    Linear1D_Col,
    Linear1D_Row,
    LinearWithGradAccum,
    PaddingEmbedding,
    PaddingLMHead,
    VocabParallelEmbedding1D,
    VocabParallelLMHead1D,
)

from .._utils import getattr_
from ..modeling.jit import get_jit_fused_dropout_add_func
from ..modeling.opt import (
    OPTPipelineForwards,
    get_jit_fused_opt_decoder_layer_forward,
    get_lm_forward_with_dist_cross_entropy,
    get_opt_decoder_forward_for_flash_attention,
    get_opt_flash_attention_forward,
)
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = [
    "OPTPolicy",
    "OPTModelPolicy",
    "OPTForCausalLMPolicy",
    "OPTForSequenceClassificationPolicy",
    "OPTForQuestionAnsweringPolicy",
]


class OPTPolicy(Policy):
    def __init__(self) -> None:
        super().__init__()

    def config_sanity_check(self):
        pass

    def preprocess(self):
        self.tie_weight = self.tie_weight_check()
        self.origin_attn_implement = self.model.config._attn_implementation
        return self.model

    def module_policy(self):
        from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoder, OPTDecoderLayer, OptFlashAttention2

        ATTN_IMPLEMENTATION = {
            "eager": OPTAttention,
            "flash_attention_2": OptFlashAttention2,
        }

        policy = {}

        attn_cls = ATTN_IMPLEMENTATION[self.model.config._attn_implementation]

        embedding_cls = None
        if self.shard_config.enable_tensor_parallelism:
            embedding_cls = VocabParallelEmbedding1D
        else:
            if self.tie_weight:
                embedding_cls = PaddingEmbedding

        if self.shard_config.enable_fused_normalization:
            norm_cls = FusedLayerNorm
        else:
            norm_cls = LayerNorm

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn("OPT doesn't support sequence parallelism now, will ignore the sequence parallelism flag.")

        use_zbv = self.pipeline_stage_manager is not None and self.pipeline_stage_manager.use_zbv

        if self.shard_config.enable_tensor_parallelism:
            assert (
                self.model.config.num_attention_heads % self.shard_config.tensor_parallel_size == 0
            ), f"The number of attention heads must be divisible by tensor parallel size."
            policy[OPTDecoderLayer] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="fc1",
                        target_module=Linear1D_Col,
                        kwargs=dict(
                            use_zbv=use_zbv,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="fc2",
                        target_module=Linear1D_Row,
                        kwargs=dict(
                            use_zbv=use_zbv,
                        ),
                    ),
                ]
            )

            policy[attn_cls] = ModulePolicyDescription(
                attribute_replacement={
                    "embed_dim": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                    "num_heads": self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="q_proj",
                        target_module=Linear1D_Col,
                        kwargs={
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="k_proj",
                        target_module=Linear1D_Col,
                        kwargs={
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="v_proj",
                        target_module=Linear1D_Col,
                        kwargs={
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="out_proj",
                        target_module=Linear1D_Row,
                        kwargs={
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                ],
            )
        elif use_zbv:
            policy[OPTDecoderLayer] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="fc1",
                        target_module=LinearWithGradAccum,
                        kwargs=dict(
                            use_zbv=use_zbv,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="fc2",
                        target_module=LinearWithGradAccum,
                        kwargs=dict(
                            use_zbv=use_zbv,
                        ),
                    ),
                ]
            )

            policy[attn_cls] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="q_proj",
                        target_module=LinearWithGradAccum,
                        kwargs={
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="k_proj",
                        target_module=LinearWithGradAccum,
                        kwargs={
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="v_proj",
                        target_module=LinearWithGradAccum,
                        kwargs={
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="out_proj",
                        target_module=LinearWithGradAccum,
                        kwargs={
                            "fp8_communication": self.shard_config.fp8_communication,
                            "use_zbv": use_zbv,
                        },
                    ),
                ],
            )
        if embedding_cls is not None:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="embed_tokens",
                    target_module=embedding_cls,
                    kwargs=(
                        {
                            "make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by,
                            "fp8_communication": self.shard_config.fp8_communication,
                        }
                        if self.shard_config.enable_tensor_parallelism
                        else {"make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by}
                    ),
                ),
                policy=policy,
                target_key=OPTDecoder,
            )

        # optimization configuration
        self.append_or_create_submodule_replacement(
            description=SubModuleReplacementDescription(
                suffix="final_layer_norm",
                target_module=norm_cls,
                ignore_if_not_exist=True,
            ),
            policy=policy,
            target_key=OPTDecoder,
        )
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="self_attn_layer_norm",
                    target_module=norm_cls,
                    ignore_if_not_exist=True,
                ),
                SubModuleReplacementDescription(
                    suffix="final_layer_norm",
                    target_module=norm_cls,
                    ignore_if_not_exist=True,
                ),
            ],
            policy=policy,
            target_key=OPTDecoderLayer,
        )

        # use flash attention
        if self.shard_config.enable_flash_attention:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_opt_flash_attention_forward(self.shard_config),
                },
                policy=policy,
                target_key=attn_cls,
            )
            if not self.shard_config.pipeline_stage_manager:
                self.append_or_create_method_replacement(
                    description={
                        "forward": get_opt_decoder_forward_for_flash_attention(self.shard_config),
                    },
                    policy=policy,
                    target_key=OPTDecoder,
                )

        # use jit fused operator
        if self.shard_config.enable_jit_fused:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_opt_decoder_layer_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=OPTDecoderLayer,
            )

        return policy

    def postprocess(self):
        return self.model

    def get_held_layers(self) -> List[nn.Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None

        if self.model.__class__.__name__ == "OPTModel":
            module = self.model.decoder
        else:
            module = self.model.model.decoder
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = stage_manager.distribute_layers(len(module.layers))
        if stage_manager.is_interleave:
            assert stage_manager.num_model_chunks is not None
            if stage_manager.is_first_stage(ignore_chunk=True):
                held_layers.append(module.embed_tokens)
                held_layers.append(module.embed_positions)
                held_layers.append(module.project_in)
            stage_indices = stage_manager.get_stage_index(layers_per_stage)
            for start_idx, end_idx in stage_indices:
                held_layers.extend(module.layers[start_idx:end_idx])
            if (stage_manager.use_zbv and stage_manager.is_first_stage(ignore_chunk=True)) or (
                not stage_manager.use_zbv and stage_manager.is_last_stage(ignore_chunk=True)
            ):
                held_layers.append(module.final_layer_norm)
                held_layers.append(module.project_out)
        else:
            if stage_manager.is_first_stage():
                held_layers.append(module.embed_tokens)
                held_layers.append(module.embed_positions)
                held_layers.append(module.project_in)
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
            held_layers.extend(module.layers[start_idx:end_idx])
            if stage_manager.is_last_stage():
                held_layers.append(module.final_layer_norm)
                held_layers.append(module.project_out)
        return held_layers

    def set_pipeline_forward(self, model_cls: nn.Module, new_forward: Callable, policy: Dict) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
        to customized forward method, and add this changing to policy."""
        if self.pipeline_stage_manager:
            stage_manager = self.pipeline_stage_manager
            if self.model.__class__.__name__ == "OPTModel":
                module = self.model.decoder
            else:
                module = self.model.model.decoder

            layers_per_stage = stage_manager.distribute_layers(len(module.layers))
            stage_index = stage_manager.get_stage_index(layers_per_stage)
            method_replacement = {
                "forward": partial(
                    new_forward,
                    stage_manager=stage_manager,
                    stage_index=stage_index,
                    shard_config=self.shard_config,
                )
            }
            self.append_or_create_method_replacement(
                description=method_replacement, policy=policy, target_key=model_cls
            )


class OPTModelPolicy(OPTPolicy):
    def module_policy(self):
        from transformers.models.opt.modeling_opt import OPTModel

        policy = super().module_policy()
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=OPTModel,
                new_forward=OPTPipelineForwards.opt_model_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[nn.Module]:
        return super().get_held_layers()

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in OPTModel."""
        return []


class OPTForCausalLMPolicy(OPTPolicy):
    def module_policy(self):
        from transformers.models.opt.modeling_opt import OPTForCausalLM

        policy = super().module_policy()
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="lm_head",
                    target_module=VocabParallelLMHead1D,
                    kwargs=dict(
                        gather_output=not self.shard_config.parallel_output,
                        make_vocab_size_divisible_by=self.shard_config.make_vocab_size_divisible_by,
                        fp8_communication=self.shard_config.fp8_communication,
                    ),
                ),
                policy=policy,
                target_key=OPTForCausalLM,
            )
            if self.shard_config.parallel_output:
                method_replacement = {"forward": get_lm_forward_with_dist_cross_entropy(self.shard_config)}
                self.append_or_create_method_replacement(
                    description=method_replacement, policy=policy, target_key=OPTForCausalLM
                )
        else:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="lm_head",
                    target_module=PaddingLMHead,
                    kwargs=dict(make_vocab_size_divisible_by=self.shard_config.make_vocab_size_divisible_by),
                ),
                policy=policy,
                target_key=OPTForCausalLM,
            )
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=OPTForCausalLM,
                new_forward=OPTPipelineForwards.opt_for_causal_lm_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_interleave:
            if (stage_manager.use_zbv and stage_manager.is_first_stage(ignore_chunk=True)) or (
                not stage_manager.use_zbv and stage_manager.is_last_stage(ignore_chunk=True)
            ):
                held_layers.append(self.model.lm_head)
        else:
            if self.pipeline_stage_manager.is_last_stage():
                held_layers.append(self.model.lm_head)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        opt_model = self.model
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            num_stages = self.pipeline_stage_manager.num_stages
            if id(opt_model.model.decoder.embed_tokens.weight) == id(opt_model.lm_head.weight):
                return [
                    {
                        0: opt_model.model.decoder.embed_tokens.weight,
                        num_stages - 1: opt_model.lm_head.weight,
                    }
                ]
        return []

    def postprocess(self):
        if self.shard_config.enable_tensor_parallelism and self.pipeline_stage_manager is None:
            binding_map = {
                "model.decoder.embed_tokens": "lm_head",
            }

            for k, v in binding_map.items():
                src_mod = getattr_(self.model, k)
                dst_mod = getattr_(self.model, v)
                dst_mod.weight = src_mod.weight

        return self.model


class OPTForSequenceClassificationPolicy(OPTPolicy):
    def module_policy(self):
        from transformers.models.opt.modeling_opt import OPTForSequenceClassification

        policy = super().module_policy()
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=OPTForSequenceClassification,
                new_forward=OPTPipelineForwards.opt_for_sequence_classification_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.score)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        "no shared params in OPTForSequenceClassification"
        return []


class OPTForQuestionAnsweringPolicy(OPTPolicy):
    def module_policy(self):
        from transformers.models.opt.modeling_opt import OPTForQuestionAnswering

        policy = super().module_policy()
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=OPTForQuestionAnswering,
                new_forward=OPTPipelineForwards.opt_for_question_answering_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_interleave:
            if (stage_manager.use_zbv and stage_manager.is_first_stage(ignore_chunk=True)) or (
                not stage_manager.use_zbv and stage_manager.is_last_stage(ignore_chunk=True)
            ):
                held_layers.append(self.model.qa_outputs)
        else:
            if self.pipeline_stage_manager.is_last_stage():
                held_layers.append(self.model.qa_outputs)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        "no shared params in OPTForSequenceClassification"
        return []

from functools import partial
from typing import Callable, Dict, List, Union

import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from colossalai.shardformer.layer import (
    FusedRMSNorm,
    Linear1D_Col,
    Linear1D_Row,
    PaddingEmbedding,
    RMSNorm,
    VocabParallelEmbedding1D,
)

from ..modeling.qwen2 import (
    Qwen2PipelineForwards,
    get_lm_forward_with_dist_cross_entropy,
    get_qwen2_flash_attention_forward,
    get_qwen2_model_forward_for_flash_attn,
)

try:
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention,
        Qwen2DecoderLayer,
        Qwen2FlashAttention2,
        Qwen2ForCausalLM,
        Qwen2ForSequenceClassification,
        Qwen2Model,
        Qwen2SdpaAttention,
    )
except ImportError:
    Qwen2ForCausalLM = "Qwen2ForCausalLM"
    Qwen2ForSequenceClassification = "Qwen2ForSequenceClassification"
    Qwen2Attention = "Qwen2Attention"
    Qwen2FlashAttention2 = "Qwen2FlashAttention2"
    Qwen2SdpaAttention = "Qwen2SdpaAttention"
    Qwen2DecoderLayer = "Qwen2DecoderLayer"
    Qwen2Model = "Qwen2Model"

from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["Qwen2Policy", "Qwen2ForCausalLMPolicy", "Qwen2ForSequenceClassificationPolicy"]


class Qwen2Policy(Policy):
    def __init__(self) -> None:
        super().__init__()
        import transformers
        from packaging.version import Version

        assert Version(transformers.__version__) >= Version(
            "4.39.1"
        ), "The Qwen2 model should run on a transformers version of 4.39.1."

    def config_sanity_check(self):
        pass

    def preprocess(self):
        self.tie_weight = self.tie_weight_check()
        self.origin_attn_implement = self.model.config._attn_implementation
        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        ATTN_IMPLEMENTATION = {
            "eager": Qwen2Attention,
            "flash_attention_2": Qwen2FlashAttention2,
            "sdpa": Qwen2SdpaAttention,
        }

        policy = {}

        attn_cls = ATTN_IMPLEMENTATION[self.origin_attn_implement]
        embedding_cls = None
        if self.shard_config.enable_tensor_parallelism:
            embedding_cls = VocabParallelEmbedding1D
        else:
            if self.tie_weight:
                embedding_cls = PaddingEmbedding
        norm_cls = FusedRMSNorm if self.shard_config.enable_fused_normalization else RMSNorm

        sp_mode = self.shard_config.sequence_parallelism_mode or None
        sp_size = self.shard_config.sequence_parallel_size or None
        sp_group = self.shard_config.sequence_parallel_process_group or None
        sp_partial_derived = sp_mode in ["split_gather", "ring"]
        if sp_mode == "all_to_all":
            decoder_attribute_replacement = {
                "num_heads": self.model.config.num_attention_heads // sp_size,
            }
            if getattr(self.model.config, "num_key_value_heads", False):
                decoder_attribute_replacement["num_key_value_heads"] = self.model.config.num_key_value_heads // sp_size

            policy[attn_cls] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
            )

        if self.shard_config.enable_tensor_parallelism:
            assert (
                self.model.config.num_attention_heads % self.shard_config.tensor_parallel_size == 0
            ), f"The number of attention heads must be divisible by tensor parallel size."
            if hasattr(self.model.config, "num_key_value_heads"):
                assert (
                    self.model.config.num_key_value_heads % self.shard_config.tensor_parallel_size == 0
                ), f"The number of key_value heads must be divisible by tensor parallel size."
            decoder_attribute_replacement = {
                "self_attn.hidden_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "self_attn.num_heads": self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
            }
            if getattr(self.model.config, "num_key_value_heads", False):
                decoder_attribute_replacement["self_attn.num_key_value_heads"] = (
                    self.model.config.num_key_value_heads // self.shard_config.tensor_parallel_size
                )

            policy[Qwen2DecoderLayer] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.q_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.k_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.v_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=Linear1D_Row,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.gate_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.up_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.down_proj",
                        target_module=Linear1D_Row,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                ],
            )

        if embedding_cls is not None:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="embed_tokens",
                    target_module=embedding_cls,
                    kwargs={"make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by},
                ),
                policy=policy,
                target_key=Qwen2Model,
            )

        # optimization configuration
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="input_layernorm",
                    target_module=norm_cls,
                    kwargs={"sp_partial_derived": sp_partial_derived},
                ),
                SubModuleReplacementDescription(
                    suffix="post_attention_layernorm",
                    target_module=norm_cls,
                    kwargs={"sp_partial_derived": sp_partial_derived},
                ),
            ],
            policy=policy,
            target_key=Qwen2DecoderLayer,
        )

        self.append_or_create_submodule_replacement(
            description=SubModuleReplacementDescription(
                suffix="norm",
                target_module=norm_cls,
                kwargs={"sp_partial_derived": sp_partial_derived},
            ),
            policy=policy,
            target_key=Qwen2Model,
        )

        if self.shard_config.enable_flash_attention or self.shard_config.enable_sequence_parallelism:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_qwen2_flash_attention_forward(self.shard_config, sp_mode, sp_size, sp_group),
                },
                policy=policy,
                target_key=attn_cls,
            )
            if self.pipeline_stage_manager is None:
                # replace qwen2 model forward method
                self.append_or_create_method_replacement(
                    description={
                        "forward": get_qwen2_model_forward_for_flash_attn(
                            self.shard_config, sp_mode, sp_size, sp_group
                        ),
                    },
                    policy=policy,
                    target_key=Qwen2Model,
                )

        return policy

    def postprocess(self):
        return self.model

    def set_pipeline_forward(self, model_cls: nn.Module, new_forward: Callable, policy: Dict) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
        to customized forward method, and add this changing to policy."""
        if self.pipeline_stage_manager is None:
            return

        stage_manager = self.pipeline_stage_manager
        if self.model.__class__.__name__ == "Qwen2Model":
            module = self.model
        else:
            module = self.model.model

        if stage_manager.is_interleave:
            layers_per_stage = stage_manager.distribute_layers(len(module.layers))
            stage_manager.stage_indices = stage_manager.get_stage_index(layers_per_stage)
            method_replacement = {
                "forward": partial(new_forward, stage_manager=stage_manager, shard_config=self.shard_config)
            }

        else:
            layers_per_stage = stage_manager.distribute_layers(len(module.layers))
            stage_index = stage_manager.get_stage_index(layers_per_stage)
            method_replacement = {
                "forward": partial(
                    new_forward, stage_manager=stage_manager, stage_index=stage_index, shard_config=self.shard_config
                )
            }
            self.append_or_create_method_replacement(
                description=method_replacement, policy=policy, target_key=model_cls
            )

        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=model_cls)

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None

        if self.model.__class__.__name__ == "Qwen2Model":
            module = self.model
        else:
            module = self.model.model

        stage_manager = self.pipeline_stage_manager

        held_layers = []
        if stage_manager.is_interleave:
            assert stage_manager.num_model_chunks is not None
            layers_per_stage = stage_manager.distribute_layers(len(module.layers))
            stage_indices = stage_manager.get_stage_index(layers_per_stage)
            if stage_manager.is_first_stage(ignore_chunk=True):
                held_layers.append(module.embed_tokens)
            for start_idx, end_idx in stage_indices:
                held_layers.extend(module.layers[start_idx:end_idx])
            if stage_manager.is_last_stage(ignore_chunk=True):
                held_layers.append(module.norm)

        else:
            layers_per_stage = stage_manager.distribute_layers(len(module.layers))
            if stage_manager.is_first_stage():
                held_layers.append(module.embed_tokens)
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
            held_layers.extend(module.layers[start_idx:end_idx])
            if stage_manager.is_last_stage():
                held_layers.append(module.norm)

        return held_layers


class Qwen2ModelPolicy(Qwen2Policy):
    def module_policy(self):
        policy = super().module_policy()

        if self.pipeline_stage_manager:
            # set None as default
            self.set_pipeline_forward(
                model_cls=Qwen2Model, new_forward=Qwen2PipelineForwards.qwen2_model_forward, policy=policy
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        held_layers = super().get_held_layers()
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in Qwen2 model"""
        return []


class Qwen2ForCausalLMPolicy(Qwen2Policy):
    def module_policy(self):
        policy = super().module_policy()
        setattr(self.shard_config, "causal_lm", True)

        if self.shard_config.enable_tensor_parallelism:
            # add a new item for casual lm
            new_item = {
                Qwen2ForCausalLM: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(suffix="lm_head", target_module=Linear1D_Col)
                    ],
                    method_replacement={"forward": get_lm_forward_with_dist_cross_entropy(self.shard_config)},
                )
            }
            policy.update(new_item)

        if self.pipeline_stage_manager:
            # set None as default
            self.set_pipeline_forward(
                model_cls=Qwen2ForCausalLM, new_forward=Qwen2PipelineForwards.qwen2_for_causal_lm_forward, policy=policy
            )

        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.lm_head)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        qwen2_model = self.model.model
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            if (
                id(qwen2_model.embed_tokens.weight) == id(self.model.lm_head.weight)
                and self.pipeline_stage_manager.num_stages > 1
            ):
                # tie weights
                return [
                    {
                        0: qwen2_model.embed_tokens.weight,
                        self.pipeline_stage_manager.num_stages - 1: self.model.lm_head.weight,
                    }
                ]
        return []


class Qwen2ForSequenceClassificationPolicy(Qwen2Policy):
    def module_policy(self):
        policy = super().module_policy()
        if self.shard_config.enable_tensor_parallelism:
            # add a new item for sequence classification
            new_item = {
                Qwen2ForSequenceClassification: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="score", target_module=Linear1D_Col, kwargs=dict(gather_output=True)
                        )
                    ]
                )
            }
            policy.update(new_item)
        # to be confirmed
        if self.pipeline_stage_manager:
            # set None as default
            self.set_pipeline_forward(
                model_cls=Qwen2ForSequenceClassification,
                new_forward=Qwen2PipelineForwards.qwen2_for_sequence_classification_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.score)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in Qwen2 for sequence classification model"""
        return []

import warnings
from functools import partial
from typing import Callable, Dict, List

from torch import Tensor, nn
from torch.nn import Module

import colossalai.shardformer.layer as col_nn

from ..modeling.falcon import (
    FalconPipelineForwards,
    build_falcon_alibi_tensor_fn,
    get_lm_forward_with_dist_cross_entropy,
    get_tp_falcon_decoder_layer_forward,
)
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["FalconPolicy"]


class FalconPolicy(Policy):
    def __init__(self) -> None:
        super().__init__()

    def config_sanity_check(self):
        pass

    def preprocess(self):
        self.tie_weight = self.tie_weight_check()
        return self.model

    def module_policy(self):
        from transformers.models.falcon.modeling_falcon import FalconDecoderLayer, FalconModel

        if not self.model.config.new_decoder_architecture and self.model.config.multi_query:
            warnings.warn(
                "Falcon doesn't support tensor parallelism when (not new_decoder_architecture and multi_query) is True, will ignore the tensor parallelism flag."
            )
            self.shard_config.enable_tensor_parallelism = False

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn("Falcon doesn't support sequence parallelism now, will ignore the sequence parallelism flag.")

        policy = {}

        embedding_cls = None
        if self.shard_config.enable_tensor_parallelism:
            embedding_cls = col_nn.VocabParallelEmbedding1D
        else:
            if self.tie_weight:
                embedding_cls = col_nn.PaddingEmbedding

        use_zbv = self.pipeline_stage_manager is not None and self.pipeline_stage_manager.use_zbv

        if self.shard_config.enable_tensor_parallelism:
            assert (
                self.model.config.num_attention_heads % self.shard_config.tensor_parallel_size == 0
            ), f"The number of attention heads must be divisible by tensor parallel size."
            assert (
                self.model.config.num_kv_heads % self.shard_config.tensor_parallel_size == 0
            ), f"The number of key_value heads must be divisible by tensor parallel size."
            attn_attribute_replacement = {
                "self_attention.hidden_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "self_attention.split_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "self_attention.num_heads": self.model.config.num_attention_heads
                // self.shard_config.tensor_parallel_size,
                "self_attention.num_kv_heads": self.model.config.num_kv_heads // self.shard_config.tensor_parallel_size,
            }

            policy[FalconDecoderLayer] = ModulePolicyDescription(
                attribute_replacement=attn_attribute_replacement,
                method_replacement={"forward": get_tp_falcon_decoder_layer_forward()},
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attention.query_key_value",
                        target_module=col_nn.Linear1D_Col,
                        kwargs=dict(
                            use_zbv=use_zbv,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attention.dense",
                        target_module=col_nn.Linear1D_Row,
                        kwargs=dict(
                            use_zbv=use_zbv,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attention.attention_dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.dense_h_to_4h",
                        target_module=col_nn.Linear1D_Col,
                        kwargs=dict(
                            use_zbv=use_zbv,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.dense_4h_to_h",
                        target_module=col_nn.Linear1D_Row,
                        kwargs=dict(
                            use_zbv=use_zbv,
                        ),
                    ),
                ],
            )

            policy[FalconModel] = ModulePolicyDescription(
                attribute_replacement={
                    "num_heads": self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
                },
                method_replacement={
                    "build_alibi_tensor": build_falcon_alibi_tensor_fn(self.shard_config.tensor_parallel_process_group)
                },
            )
        elif use_zbv:
            policy[FalconDecoderLayer] = ModulePolicyDescription(
                method_replacement={"forward": get_tp_falcon_decoder_layer_forward()},
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attention.query_key_value",
                        target_module=col_nn.LinearWithGradAccum,
                        kwargs=dict(
                            use_zbv=use_zbv,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attention.dense",
                        target_module=col_nn.LinearWithGradAccum,
                        kwargs=dict(
                            use_zbv=use_zbv,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attention.attention_dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.dense_h_to_4h",
                        target_module=col_nn.LinearWithGradAccum,
                        kwargs=dict(
                            use_zbv=use_zbv,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.dense_4h_to_h",
                        target_module=col_nn.LinearWithGradAccum,
                        kwargs=dict(
                            use_zbv=use_zbv,
                        ),
                    ),
                ],
            )

        if embedding_cls is not None:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="word_embeddings",
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
                ],
                policy=policy,
                target_key=FalconModel,
            )

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            # handle falcon model
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="ln_f",
                        target_module=col_nn.FusedLayerNorm,
                    ),
                ],
                policy=policy,
                target_key=FalconModel,
            )

            # handle falcon decoder layer
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="ln_attn", target_module=col_nn.FusedLayerNorm, ignore_if_not_exist=True
                    ),
                    SubModuleReplacementDescription(
                        suffix="ln_mlp", target_module=col_nn.FusedLayerNorm, ignore_if_not_exist=True
                    ),
                    SubModuleReplacementDescription(
                        suffix="input_layernorm", target_module=col_nn.FusedLayerNorm, ignore_if_not_exist=True
                    ),
                    SubModuleReplacementDescription(
                        suffix="post_attention_layernorm", target_module=col_nn.FusedLayerNorm, ignore_if_not_exist=True
                    ),
                ],
                policy=policy,
                target_key=FalconDecoderLayer,
            )

        if self.shard_config.enable_flash_attention:
            warnings.warn("Falcon doesn't support flash attention now, fallback to transformers attention.")

        return policy

    def postprocess(self):
        return self.model

    def set_pipeline_forward(self, model_cls: nn.Module, new_forward: Callable, policy: Dict) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
        to customized forward method, and add this changing to policy."""
        if self.pipeline_stage_manager:
            stage_manager = self.pipeline_stage_manager
            if self.model.__class__.__name__ == "FalconModel":
                module = self.model
            else:
                module = self.model.transformer

            layers_per_stage = stage_manager.distribute_layers(len(module.h))
            stage_index = stage_manager.get_stage_index(layers_per_stage)
            method_replacement = {
                "forward": partial(
                    new_forward, stage_manager=stage_manager, stage_index=stage_index, shard_config=self.shard_config
                )
            }
            self.append_or_create_method_replacement(
                description=method_replacement, policy=policy, target_key=model_cls
            )

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None
        if self.model.__class__.__name__ == "FalconModel":
            module = self.model
        else:
            module = self.model.transformer
        stage_manager = self.pipeline_stage_manager
        held_layers = []
        if stage_manager.is_interleave:
            assert stage_manager.num_model_chunks is not None
            layers_per_stage = stage_manager.distribute_layers(len(module.h))
            stage_indices = stage_manager.get_stage_index(layers_per_stage)
            if stage_manager.is_first_stage(ignore_chunk=True):
                held_layers.append(module.word_embeddings)
            for start_idx, end_idx in stage_indices:
                held_layers.extend(module.h[start_idx:end_idx])
            if (stage_manager.use_zbv and stage_manager.is_first_stage(ignore_chunk=True)) or (
                not stage_manager.use_zbv and stage_manager.is_last_stage(ignore_chunk=True)
            ):
                held_layers.append(module.ln_f)
        else:
            layers_per_stage = stage_manager.distribute_layers(len(module.h))
            if stage_manager.is_first_stage():
                held_layers.append(module.word_embeddings)
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
            held_layers.extend(module.h[start_idx:end_idx])
            if stage_manager.is_last_stage():
                held_layers.append(module.ln_f)

        return held_layers


class FalconModelPolicy(FalconPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()

        from transformers.models.falcon.modeling_falcon import FalconModel

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=FalconModel, new_forward=FalconPipelineForwards.falcon_model_forward, policy=policy
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        held_layers = super().get_held_layers()
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """no shared params in falcon model"""
        return []


class FalconForCausalLMPolicy(FalconPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.falcon.modeling_falcon import FalconForCausalLM

        policy = super().module_policy()

        # handle tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="lm_head",
                    target_module=col_nn.VocabParallelLMHead1D,
                    kwargs=dict(
                        gather_output=not self.shard_config.parallel_output,
                        make_vocab_size_divisible_by=self.shard_config.make_vocab_size_divisible_by,
                    ),
                ),
                policy=policy,
                target_key=FalconForCausalLM,
            )
            if self.shard_config.parallel_output:
                method_replacement = {"forward": get_lm_forward_with_dist_cross_entropy(self.shard_config)}
                self.append_or_create_method_replacement(
                    description=method_replacement, policy=policy, target_key=FalconForCausalLM
                )

        else:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="lm_head",
                    target_module=col_nn.PaddingLMHead,
                    kwargs=dict(make_vocab_size_divisible_by=self.shard_config.make_vocab_size_divisible_by),
                ),
                policy=policy,
                target_key=FalconForCausalLM,
            )

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=FalconForCausalLM,
                new_forward=FalconPipelineForwards.falcon_for_causal_lm_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_interleave:
            if (stage_manager.use_zbv and stage_manager.is_first_stage(ignore_chunk=True)) or (
                not stage_manager.use_zbv and stage_manager.is_last_stage(ignore_chunk=True)
            ):
                held_layers.append(self.model.lm_head)
        else:
            if stage_manager.is_last_stage():
                held_layers.append(self.model.lm_head)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        falcon_model = self.model
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            if id(falcon_model.transformer.word_embeddings.weight) == id(falcon_model.lm_head.weight):
                # tie weights
                return [
                    {
                        0: falcon_model.transformer.word_embeddings.weight,
                        self.pipeline_stage_manager.num_stages - 1: falcon_model.lm_head.weight,
                    }
                ]
        return []


class FalconForSequenceClassificationPolicy(FalconPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.falcon.modeling_falcon import FalconForSequenceClassification

        policy = super().module_policy()

        use_zbv = self.pipeline_stage_manager is not None and self.pipeline_stage_manager.use_zbv

        # handle tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="score", target_module=col_nn.Linear1D_Col, kwargs=dict(gather_output=True, use_zbv=use_zbv)
                ),
                policy=policy,
                target_key=FalconForSequenceClassification,
            )
        elif use_zbv:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="score",
                    target_module=col_nn.LinearWithGradAccum,
                    kwargs=dict(gather_output=True, use_zbv=use_zbv),
                ),
                policy=policy,
                target_key=FalconForSequenceClassification,
            )

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=FalconForSequenceClassification,
                new_forward=FalconPipelineForwards.falcon_for_sequence_classification_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_interleave:
            if (stage_manager.use_zbv and stage_manager.is_first_stage(ignore_chunk=True)) or (
                not stage_manager.use_zbv and stage_manager.is_last_stage(ignore_chunk=True)
            ):
                held_layers.append(self.model.score)
        else:
            if stage_manager.is_last_stage():
                held_layers.append(self.model.score)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in falcon for sequence classification model"""
        return []


class FalconForTokenClassificationPolicy(FalconPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.falcon.modeling_falcon import FalconForTokenClassification

        policy = super().module_policy()

        use_zbv = self.pipeline_stage_manager is not None and self.pipeline_stage_manager.use_zbv

        # handle tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="classifier",
                        target_module=col_nn.Linear1D_Col,
                        kwargs=dict(gather_output=True, use_zbv=use_zbv),
                    ),
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=col_nn.DropoutForReplicatedInput,
                    ),
                ],
                policy=policy,
                target_key=FalconForTokenClassification,
            )
        elif use_zbv:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="classifier",
                        target_module=col_nn.LinearWithGradAccum,
                        kwargs=dict(gather_output=True, use_zbv=use_zbv),
                    ),
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=col_nn.DropoutForReplicatedInput,
                    ),
                ],
                policy=policy,
                target_key=FalconForTokenClassification,
            )
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=FalconForTokenClassification,
                new_forward=FalconPipelineForwards.falcon_for_token_classification_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_interleave:
            if (stage_manager.use_zbv and stage_manager.is_first_stage(ignore_chunk=True)) or (
                not stage_manager.use_zbv and stage_manager.is_last_stage(ignore_chunk=True)
            ):
                held_layers.append(self.model.dropout)
                held_layers.append(self.model.classifier)
        else:
            if stage_manager.is_last_stage():
                held_layers.append(self.model.dropout)
                held_layers.append(self.model.classifier)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in falcon for token classification model"""
        return []


class FalconForQuestionAnsweringPolicy(FalconPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.falcon.modeling_falcon import FalconForQuestionAnswering

        policy = super().module_policy()

        use_zbv = self.pipeline_stage_manager is not None and self.pipeline_stage_manager.use_zbv

        # handle tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="qa_outputs",
                    target_module=col_nn.Linear1D_Col,
                    kwargs=dict(gather_output=True, use_zbv=use_zbv),
                ),
                policy=policy,
                target_key=FalconForQuestionAnswering,
            )
        elif use_zbv:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="qa_outputs",
                    target_module=col_nn.Linear1D_Col,
                    kwargs=dict(gather_output=True, use_zbv=use_zbv),
                ),
                policy=policy,
                target_key=FalconForQuestionAnswering,
            )
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=FalconForQuestionAnswering,
                new_forward=FalconPipelineForwards.falcon_for_question_answering_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        held_layers = super().get_held_layers()
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_interleave:
            if (stage_manager.use_zbv and stage_manager.is_first_stage(ignore_chunk=True)) or (
                not stage_manager.use_zbv and stage_manager.is_last_stage(ignore_chunk=True)
            ):
                held_layers.append(self.model.qa_outputs)
        else:
            if stage_manager.is_last_stage():
                held_layers.append(self.model.qa_outputs)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in falcon for question answering model"""
        return []

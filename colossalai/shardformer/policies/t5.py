from __future__ import annotations

import warnings
from functools import partial
from typing import Callable, Dict, List, Tuple

import numpy as np
from torch import Tensor, nn

from colossalai.shardformer.layer import (
    DropoutForParallelInput,
    Embedding1D,
    FusedRMSNorm,
    Linear1D_Col,
    Linear1D_Row,
    PaddingEmbedding,
    PaddingLMHead,
    RMSNorm,
    VocabParallelEmbedding1D,
    VocabParallelLMHead1D,
)
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription

from ..modeling.jit import get_jit_fused_dropout_add_func
from ..modeling.t5 import (
    T5PipelineForwards,
    get_jit_fused_T5_layer_ff_forward,
    get_t5_flash_attention_forward,
    get_T5_layer_cross_attention_forward,
    get_T5_layer_self_attention_forward,
)
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = [
    "distribute_t5_layers",
    "T5ModelPolicy",
    "T5ForConditionalGenerationPolicy",
    "T5EncoderPolicy",
    "T5ForTokenClassificationPolicy",
]


class T5BasePolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        self.tie_weight = self.tie_weight_check()
        return self.model

    def module_policy(self):
        from transformers.models.t5.modeling_t5 import (
            T5Attention,
            T5DenseActDense,
            T5DenseGatedActDense,
            T5LayerCrossAttention,
            T5LayerFF,
            T5LayerSelfAttention,
            T5Stack,
        )

        policy = {}

        embedding_cls = None
        if self.shard_config.enable_tensor_parallelism:
            embedding_cls = VocabParallelEmbedding1D
        else:
            if self.tie_weight:
                embedding_cls = PaddingEmbedding

        if self.shard_config.enable_fused_normalization:
            norm_cls = FusedRMSNorm
        else:
            norm_cls = RMSNorm

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn("T5 doesn't support sequence parallelism now, will ignore the sequence parallelism flag.")

        if self.shard_config.enable_tensor_parallelism:
            assert (
                self.model.config.num_heads % self.shard_config.tensor_parallel_size == 0
            ), f"The number of attention heads must be divisible by tensor parallel size."
            policy[T5Stack] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=DropoutForParallelInput,
                    ),
                ]
            )
            policy[T5LayerSelfAttention] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=DropoutForParallelInput,
                    ),
                ]
            )
            policy[T5LayerCrossAttention] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=DropoutForParallelInput,
                    )
                ]
            )
            policy[T5Attention] = ModulePolicyDescription(
                attribute_replacement={
                    "d_model": self.model.config.d_model // self.shard_config.tensor_parallel_size,
                    "n_heads": self.model.config.num_heads // self.shard_config.tensor_parallel_size,
                    "inner_dim": self.model.config.num_heads
                    * self.model.config.d_kv
                    // self.shard_config.tensor_parallel_size,
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="q",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="k",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="v",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="o",
                        target_module=Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="relative_attention_bias",
                        target_module=Embedding1D,
                        kwargs=dict(gather_output=False),
                        ignore_if_not_exist=True,
                    ),
                ],
            )
            policy[T5LayerFF] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=DropoutForParallelInput,
                    ),
                ]
            )
            policy[T5DenseGatedActDense] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="wi_0 ",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="wi_1",
                        target_module=Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="wo", target_module=Linear1D_Col, kwargs=dict(gather_output=True)
                    ),
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=DropoutForParallelInput,
                    ),
                ]
            )
            policy[T5DenseActDense] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="wi",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="wo",
                        target_module=Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=DropoutForParallelInput,
                    ),
                ]
            )

        if embedding_cls is not None:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="embed_tokens",
                    target_module=embedding_cls,
                    kwargs={"make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by},
                ),
                policy=policy,
                target_key=T5Stack,
            )

        # optimization configuration
        self.append_or_create_submodule_replacement(
            description=SubModuleReplacementDescription(
                suffix="layer_norm",
                target_module=norm_cls,
            ),
            policy=policy,
            target_key=T5LayerFF,
        )
        self.append_or_create_submodule_replacement(
            description=SubModuleReplacementDescription(suffix="layer_norm", target_module=norm_cls),
            policy=policy,
            target_key=T5LayerSelfAttention,
        )
        self.append_or_create_submodule_replacement(
            description=SubModuleReplacementDescription(suffix="layer_norm", target_module=norm_cls),
            policy=policy,
            target_key=T5LayerCrossAttention,
        )
        self.append_or_create_submodule_replacement(
            description=SubModuleReplacementDescription(suffix="final_layer_norm", target_module=norm_cls),
            policy=policy,
            target_key=T5Stack,
        )

        # use flash attention
        if self.shard_config.enable_flash_attention:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_t5_flash_attention_forward(),
                },
                policy=policy,
                target_key=T5Attention,
            )

        # use jit operator
        if self.shard_config.enable_jit_fused:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_T5_layer_ff_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=T5LayerFF,
            )
            self.append_or_create_method_replacement(
                description={
                    "forward": get_T5_layer_self_attention_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=T5LayerSelfAttention,
            )
            self.append_or_create_method_replacement(
                description={
                    "forward": get_T5_layer_cross_attention_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=T5LayerCrossAttention,
            )

        return policy

    def postprocess(self):
        return self.model

    def distribute_t5_layers(
        self, num_encoder_layers: int, num_decoder_layers: int, num_stages: int
    ) -> Tuple[List[int], int]:
        """
        Distribute t5 layers into stages when pipeline parallel is used.
        Return the layer distribution as a list and the starting stage of decoder.
        If decoder doesn't exist, returned decoder starting stage is set to num_encoder_layers.
        """
        stage_manager = self.pipeline_stage_manager
        assert stage_manager is not None, "Pipeline stage manager is not set."

        # number of encoder layers must be a positive integer
        if num_encoder_layers <= 0:
            raise ValueError("The number of encoder layers for T5 must be a positive integer.")

        # number of layers should be large enough to fill in every stage
        if num_encoder_layers + num_decoder_layers < num_stages:
            raise ValueError("The total number of layers can't be smaller than number of stages.")

        # in the case of T5EncoderModel, set decoder starting stage to num_stages since it doesn't exist
        if num_decoder_layers == 0:
            return stage_manager.distribute_layers(num_encoder_layers, num_stages), num_stages

        # the number of stages distributed between encoder and decoder is optimized in this way:
        # num_encoder_stages = argmin(abs(num_encoder_layers / encoder_stages - num_decoder_layers / decoder_stages))
        #                   s.t. num_encoder_stages + num_decoder_stages = num_stages, num_encoder_stages >= 1, num_decoder_stages >= 1
        def objective(num_encoder_stages):
            return abs(num_encoder_layers / num_encoder_stages - num_decoder_layers / (num_stages - num_encoder_stages))

        num_encoder_stages = np.argmin([objective(i) for i in range(1, num_stages)]) + 1
        num_decoder_stages = num_stages - num_encoder_stages

        encoder_distribution = stage_manager.distribute_layers(num_encoder_layers, num_encoder_stages)
        decoder_distribution = stage_manager.distribute_layers(num_decoder_layers, num_decoder_stages)
        return encoder_distribution + decoder_distribution, num_encoder_stages

    def get_t5_stage_index(
        self, layers_per_stage: List[int], stage: int, decoder_starting_stage: int
    ) -> Tuple[int, int]:
        """
        Input the distribution of layers among stages, the current stage and the first stage of decoder.
        Return the starting/ending idx of layers in encoder/decoder
        """
        stage_manager = self.pipeline_stage_manager
        assert stage_manager is not None, "Pipeline stage manager is not set."

        if stage < decoder_starting_stage:
            return stage_manager.get_stage_index(layers_per_stage[:decoder_starting_stage], stage)
        else:
            return stage_manager.get_stage_index(
                layers_per_stage[decoder_starting_stage:], stage - decoder_starting_stage
            )

    def get_held_layers(self) -> List[nn.Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None
        stage_manager = self.pipeline_stage_manager

        if self.model.__class__.__name__ == "T5ForTokenClassification":
            model = self.model.transformer
        else:
            model = self.model

        encoder = model.encoder
        decoder = getattr(model, "decoder", None)

        num_encoder_layers = len(encoder.block)
        num_decoder_layers = len(decoder.block) if decoder else 0

        held_layers = []
        layers_per_stage, decoder_starting_stage = self.distribute_t5_layers(
            num_encoder_layers, num_decoder_layers, stage_manager.num_stages
        )
        start_idx, end_idx = self.get_t5_stage_index(layers_per_stage, stage_manager.stage, decoder_starting_stage)

        if stage_manager.stage < decoder_starting_stage:
            # current stage is in t5's encoder
            if stage_manager.is_first_stage():
                held_layers.append(model.shared)
                held_layers.append(encoder.embed_tokens)
                held_layers.append(encoder.dropout)
            if stage_manager.stage == decoder_starting_stage - 1:
                held_layers.append(encoder.final_layer_norm)
                held_layers.append(encoder.dropout)
            held_layers.extend(encoder.block[start_idx:end_idx])
        else:
            # current stage is in t5's decoder
            if stage_manager.stage == decoder_starting_stage:
                held_layers.append(decoder.embed_tokens)
                held_layers.append(decoder.dropout)
            if stage_manager.is_last_stage():
                held_layers.append(decoder.final_layer_norm)
                held_layers.append(decoder.dropout)
            held_layers.extend(decoder.block[start_idx:end_idx])
        return held_layers

    def set_pipeline_forward(self, model_cls: nn.Module, new_forward: Callable, policy: Dict) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
        to customized forward method, and add this changing to policy."""
        if not self.pipeline_stage_manager:
            raise ValueError("set_pipeline_forward method can only be called when pipeline parallel is enabled.")
        stage_manager = self.pipeline_stage_manager

        if self.model.__class__.__name__ == "T5ForTokenClassification":
            encoder = self.model.transformer.encoder
        else:
            encoder = self.model.encoder

        decoder = getattr(self.model, "decoder", None)

        num_encoder_layers = len(encoder.block)
        num_decoder_layers = len(decoder.block) if decoder else 0

        layers_per_stage, decoder_starting_stage = self.distribute_t5_layers(
            num_encoder_layers, num_decoder_layers, stage_manager.num_stages
        )
        stage_index = self.get_t5_stage_index(layers_per_stage, stage_manager.stage, decoder_starting_stage)

        method_replacement = {
            "forward": partial(
                new_forward,
                stage_manager=stage_manager,
                stage_index=stage_index,
                decoder_starting_stage=decoder_starting_stage,
            )
        }
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=model_cls)


class T5ModelPolicy(T5BasePolicy):
    def module_policy(self):
        from transformers import T5Model

        policy = super().module_policy()

        embedding_cls = None
        if self.shard_config.enable_tensor_parallelism:
            embedding_cls = VocabParallelEmbedding1D
        else:
            if self.tie_weight:
                embedding_cls = PaddingEmbedding

        if embedding_cls is not None:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="shared",
                    target_module=embedding_cls,
                    kwargs={"make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by},
                ),
                policy=policy,
                target_key=T5Model,
            )
        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(model_cls=T5Model, new_forward=T5PipelineForwards.t5_model_forward, policy=policy)

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        return super().get_held_layers()

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        module = self.model
        stage_manager = self.pipeline_stage_manager
        if stage_manager is not None and stage_manager.num_stages > 1:
            _, decoder_starting_stage = self.distribute_t5_layers(
                len(module.encoder.block), len(module.decoder.block), stage_manager.num_stages
            )

            if id(module.decoder.embed_tokens.weight) == id(module.shared.weight):
                return [{0: module.shared.weight, decoder_starting_stage: module.decoder.embed_tokens.weight}]
        return []


class T5ForConditionalGenerationPolicy(T5BasePolicy):
    def module_policy(self):
        from transformers import T5ForConditionalGeneration

        policy = super().module_policy()

        embedding_cls = None
        if self.shard_config.enable_tensor_parallelism:
            embedding_cls = VocabParallelEmbedding1D
        else:
            if self.tie_weight:
                embedding_cls = PaddingEmbedding

        if embedding_cls is not None:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="shared",
                    target_module=embedding_cls,
                    kwargs={"make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by},
                ),
                policy=policy,
                target_key=T5ForConditionalGeneration,
            )

        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="lm_head",
                    target_module=VocabParallelLMHead1D,
                    kwargs={
                        "gather_output": True,
                        "make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by,
                    },
                ),
                policy=policy,
                target_key=T5ForConditionalGeneration,
            )
        else:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="lm_head",
                    target_module=PaddingLMHead,
                    kwargs={"make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by},
                ),
                policy=policy,
                target_key=T5ForConditionalGeneration,
            )

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(
                model_cls=T5ForConditionalGeneration,
                new_forward=T5PipelineForwards.t5_for_conditional_generation_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.lm_head)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        module = self.model
        stage_manager = self.pipeline_stage_manager
        if stage_manager is not None and stage_manager.num_stages > 1:
            _, decoder_starting_stage = self.distribute_t5_layers(
                len(module.encoder.block), len(module.decoder.block), stage_manager.num_stages
            )

            shared_params = []
            shared_embedding = {}
            if id(module.decoder.embed_tokens.weight) == id(module.shared.weight):
                shared_embedding[0] = module.shared.weight
                shared_embedding[decoder_starting_stage] = module.decoder.embed_tokens.weight

            if id(module.lm_head.weight) == id(module.shared.weight):
                shared_embedding[0] = module.shared.weight
                shared_embedding[stage_manager.num_stages - 1] = module.lm_head.weight

            if len(shared_embedding) > 0:
                shared_params.append(shared_embedding)

            return shared_params

        return []


class T5EncoderPolicy(T5BasePolicy):
    def module_policy(self):
        from transformers import T5EncoderModel

        policy = super().module_policy()

        embedding_cls = None
        if self.shard_config.enable_tensor_parallelism:
            embedding_cls = VocabParallelEmbedding1D
        else:
            if self.tie_weight:
                embedding_cls = PaddingEmbedding

        if embedding_cls is not None:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="shared",
                    target_module=embedding_cls,
                    kwargs={"make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by},
                ),
                policy=policy,
                target_key=T5EncoderModel,
            )

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(
                model_cls=T5EncoderModel, new_forward=T5PipelineForwards.t5_encoder_model_forward, policy=policy
            )
        return policy

    def get_held_layers(self) -> List[nn.Module]:
        return super().get_held_layers()

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        return []


class T5ForTokenClassificationPolicy(T5EncoderPolicy):
    def module_policy(self):
        from transformers.models.t5.modeling_t5 import T5ForTokenClassification

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            addon_module = {
                T5ForTokenClassification: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="dropout",
                            target_module=DropoutForParallelInput,
                        )
                    ]
                )
            }
            policy.update(addon_module)
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=T5ForTokenClassification,
                new_forward=T5PipelineForwards.t5_for_token_classification_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        """
        get pipeline layers for current stage
        """
        held_layers = super().get_held_layers()
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.dropout)
            held_layers.append(self.model.classifier)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        # no shared params for sequence classification model
        return []

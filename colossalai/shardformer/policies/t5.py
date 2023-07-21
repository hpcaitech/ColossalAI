from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

from torch import Tensor, nn

from colossalai.shardformer.layer import (
    DropoutForParallelInput,
    Embedding1D,
    FusedRMSNorm,
    Linear1D_Col,
    Linear1D_Row,
    VocabParallelEmbedding1D,
)
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription

from .._utils import getattr_, setattr_
from ..modeling.t5 import T5PipelineForwards
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["distribute_t5_layers", "T5ModelPolicy", "T5ForConditionalGenerationPolicy", "T5EncoderPolicy"]


class T5BasePolicy(Policy):

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

        if self.shard_config.enable_tensor_parallelism:
            policy[T5Stack] = ModulePolicyDescription(sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="dropout",
                    target_module=DropoutForParallelInput,
                ),
                SubModuleReplacementDescription(
                    suffix="embed_tokens",
                    target_module=VocabParallelEmbedding1D,
                )
            ])
            policy[T5LayerSelfAttention] = ModulePolicyDescription(sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="dropout",
                    target_module=DropoutForParallelInput,
                ),
            ])
            policy[T5LayerCrossAttention] = ModulePolicyDescription(sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="dropout",
                    target_module=DropoutForParallelInput,
                )
            ])
            policy[T5Attention] = ModulePolicyDescription(attribute_replacement={
                "d_model":
                    self.model.config.d_model // self.shard_config.tensor_parallel_size,
                "n_heads":
                    self.model.config.num_heads // self.shard_config.tensor_parallel_size,
                "inner_dim":
                    self.model.config.num_heads * self.model.config.d_kv // self.shard_config.tensor_parallel_size
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
                                                                  ignore_if_not_exist=True)
                                                          ])
            policy[T5LayerFF] = ModulePolicyDescription(sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="dropout",
                    target_module=DropoutForParallelInput,
                ),
            ])
            policy[T5DenseGatedActDense] = ModulePolicyDescription(sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="wi_0 ",
                    target_module=Linear1D_Col,
                ),
                SubModuleReplacementDescription(
                    suffix="wi_1",
                    target_module=Linear1D_Row,
                ),
                SubModuleReplacementDescription(
                    suffix="wo", target_module=Linear1D_Col, kwargs=dict(gather_output=True)),
                SubModuleReplacementDescription(
                    suffix="dropout",
                    target_module=DropoutForParallelInput,
                )
            ])
            policy[T5DenseActDense] = ModulePolicyDescription(sub_module_replacement=[
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
                )
            ])

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="layer_norm",
                target_module=FusedRMSNorm,
            ),
                                                        policy=policy,
                                                        target_key=T5LayerFF)
            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="layer_norm",
                target_module=FusedRMSNorm,
            ),
                                                        policy=policy,
                                                        target_key=T5LayerFF)
            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="layer_norm", target_module=FusedRMSNorm),
                                                        policy=policy,
                                                        target_key=T5LayerSelfAttention)
            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="layer_norm", target_module=FusedRMSNorm),
                                                        policy=policy,
                                                        target_key=T5LayerCrossAttention)
            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="final_layer_norm", target_module=FusedRMSNorm),
                                                        policy=policy,
                                                        target_key=T5Stack)
        return policy

    def postprocess(self):
        return self.model

    @staticmethod
    def distribute_t5_layers(num_encoder_layers: int, num_decoder_layers: int,
                             num_stages: int) -> Tuple[List[int], int]:
        """
        Distribute t5 layers into stages when pipeline parallel is used.
        Return the layer distribution as a list and the starting stage of decoder.
        If decoder doesn't exist, returned decoder starting stage is set to num_encoder_layers.
        """

        # number of encoder layers must be a positive integer
        if num_encoder_layers <= 0:
            raise ValueError("The number of encoder layers for T5 must be a positive integer.")

        # number of layers should be large enough to fill in every stage
        if num_encoder_layers + num_decoder_layers < num_stages:
            raise ValueError("The total number of layers can't be smaller than number of stages.")

        # in the case of T5EncoderModel, set decoder starting stage to num_stages since it doesn't exist
        if num_decoder_layers == 0:
            return Policy.distribute_layers(num_encoder_layers, num_stages), num_stages

        # the number of stages distributed between encoder and decoder is optmized in this way:
        # num_encoder_stages = argmin(abs(num_encoder_layers / encoder_stages - num_decoder_layers / decoder_stages))
        #                   s.t. num_encoder_stages + num_decoder_stages = num_stages, num_encoder_stages >= 1, num_decoder_stages >= 1
        def objective(num_encoder_stages):
            return abs(num_encoder_layers / num_encoder_stages - num_decoder_layers / (num_stages - num_encoder_stages))

        num_encoder_stages = 0
        optimal_diff = 2**31 - 1
        for i in range(1, num_stages):
            attempt = objective(i)
            if attempt < optimal_diff:
                num_encoder_stages = i
                optimal_diff = attempt
        num_decoder_stages = num_stages - num_encoder_stages

        encoder_distribution = Policy.distribute_layers(num_encoder_layers, num_encoder_stages)
        decoder_distribution = Policy.distribute_layers(num_decoder_layers, num_decoder_stages)
        return encoder_distribution + decoder_distribution, num_encoder_stages

    @staticmethod
    def get_t5_stage_index(layers_per_stage: List[int], stage: int,
                           decoder_starting_stage: int) -> Tuple[bool, int, int]:
        """
        Input the distribution of layers among stages, the current stage and the first stage of decoder.
        Return the starting/ending idx of layers in encoder/decoder
        """
        if stage < decoder_starting_stage:
            return Policy.get_stage_index(layers_per_stage[:decoder_starting_stage], stage)
        else:
            return Policy.get_stage_index(layers_per_stage[decoder_starting_stage:], stage - decoder_starting_stage)

    def get_held_layers(self) -> List[nn.Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None
        stage_manager = self.pipeline_stage_manager

        model = self.model
        encoder = self.model.encoder
        decoder = self.model.__dict__.get('decoder', None)

        num_encoder_layers = len(encoder.block)
        num_decoder_layers = len(decoder.block) if decoder else 0

        held_layers = []
        layers_per_stage, decoder_starting_stage = T5BasePolicy.distribute_t5_layers(
            num_encoder_layers, num_decoder_layers, stage_manager.num_stages)
        start_idx, end_idx = T5BasePolicy.get_t5_stage_index(layers_per_stage, stage_manager.stage,
                                                             decoder_starting_stage)

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

        encoder = self.model.encoder
        decoder = self.model.__dict__.get('decoder', None)

        num_encoder_layers = len(encoder.block)
        num_decoder_layers = len(decoder.block) if decoder else 0

        layers_per_stage, decoder_starting_stage = T5BasePolicy.distribute_t5_layers(
            num_encoder_layers, num_decoder_layers, stage_manager.num_stages)
        stage_index = T5BasePolicy.get_t5_stage_index(layers_per_stage, stage_manager.stage, decoder_starting_stage)

        method_replacement = {
            'forward':
                partial(new_forward,
                        stage_manager=stage_manager,
                        stage_index=stage_index,
                        decoder_starting_stage=decoder_starting_stage)
        }
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=model_cls)


class T5ModelPolicy(T5BasePolicy):

    def module_policy(self):
        from transformers import T5Model
        base_policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="shared",
                target_module=VocabParallelEmbedding1D,
            ),
                                                        policy=base_policy,
                                                        target_key=T5Model)
        return base_policy

    def postprocess(self):
        if self.shard_config.enable_tensor_parallelism:
            binding_map = {"shared.weight": ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]}
            for k, v in binding_map.items():
                src = getattr_(self.model, k)
                for dst in v:
                    setattr_(self.model, dst, src)
        return self.model


class T5ForConditionalGenerationPolicy(T5BasePolicy):

    def module_policy(self):
        from transformers import T5ForConditionalGeneration

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(description=[
                SubModuleReplacementDescription(
                    suffix="shared",
                    target_module=VocabParallelEmbedding1D,
                ),
                SubModuleReplacementDescription(suffix="lm_head",
                                                target_module=Linear1D_Col,
                                                kwargs=dict(gather_output=True))
            ],
                                                        policy=policy,
                                                        target_key=T5ForConditionalGeneration)
        return policy

    def postprocess(self):
        super().postprocess()
        if self.shard_config.enable_tensor_parallelism and self.pipeline_stage_manager is None:
            binding_map = {
                "shared.weight": ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
            }
            for k, v in binding_map.items():
                src = getattr_(self.model, k)
                for dst in v:
                    setattr_(self.model, dst, src)

        return self.model


class T5EncoderPolicy(T5BasePolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers import T5EncoderModel

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="shared",
                target_module=VocabParallelEmbedding1D,
            ),
                                                        policy=policy,
                                                        target_key=T5EncoderModel)

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(model_cls=T5EncoderModel,
                                      new_forward=T5PipelineForwards.t5_encoder_model_forward,
                                      policy=policy)
        return policy

    def get_held_layers(self) -> List[nn.Module]:
        return super().get_held_layers()

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        return []

    def postprocess(self):
        if self.shard_config.enable_tensor_parallelism:
            binding_map = {"shared.weight": ["encoder.embed_tokens.weight"]}
            for k, v in binding_map.items():
                src = getattr_(self.model, k)
                for dst in v:
                    setattr_(self.model, dst, src)
        return self.model

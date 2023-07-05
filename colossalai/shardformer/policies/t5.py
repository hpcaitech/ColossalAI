from colossalai.shardformer.layer import (
    DropoutForParallelInput,
    Embedding1D,
    FusedRMSNorm,
    Linear1D_Col,
    Linear1D_Row,
    VocabParallelEmbedding1D,
)
from colossalai.shardformer.policies.basepolicy import ModulePolicyDescription

from .._utils import getattr_, setattr_
from .basepolicy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["T5ModelPolicy", "T5ForConditionalGenerationPolicy", "T5EncoderPolicy"]


class T5BasePolicy(Policy):

    def config_sanity_check(self):
        pass

    def preprocess(self):
        # reshape the embedding layer
        r"""
        Reshape the Embedding layer to make the embedding dimension divisible by world_size
        """
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
                    target_module=Embedding1D,
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
                    suffix="wi_0",
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
        binding_map = [["shared", "encoder.embed_tokens"], ["shared", "decoder.embed_tokens"]]

        for k, v in binding_map:
            mod = getattr_(self.model, k)
            setattr_(self.model, v, mod)
        return self.model


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

        binding_map = {"shared": "lm_head"}

        for k, v in binding_map.items():
            src_mod = getattr_(self.model, k)
            dst_mod = getattr_(self.model, v)
            dst_mod.weight = src_mod.weight

        return self.model


class T5EncoderPolicy(T5BasePolicy):

    def module_policy(self):
        from transformers import T5EncoderModel

        base_policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="shared",
                target_module=VocabParallelEmbedding1D,
            ),
                                                        policy=base_policy,
                                                        target_key=T5EncoderModel)
        return base_policy

    def postprocess(self):
        binding_map = [
            ["shared", "encoder.embed_tokens"],
        ]

        for k, v in binding_map:
            mod = getattr_(self.model, k)
            setattr_(self.model, v, mod)
        return self.model

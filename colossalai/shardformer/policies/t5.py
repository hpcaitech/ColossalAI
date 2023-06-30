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

        base_policy = {
            T5Stack:
                ModulePolicyDescription(attribute_replacement={},
                                        param_replacement=[],
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(
                                                suffix="dropout",
                                                target_module=DropoutForParallelInput,
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="embed_tokens",
                                                target_module=Embedding1D,
                                            )
                                        ]),
            T5LayerSelfAttention:
                ModulePolicyDescription(attribute_replacement={},
                                        param_replacement=[],
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(
                                                suffix="dropout",
                                                target_module=DropoutForParallelInput,
                                            ),
                                        ]),
            T5LayerCrossAttention:
                ModulePolicyDescription(attribute_replacement={},
                                        param_replacement=[],
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(
                                                suffix="dropout",
                                                target_module=DropoutForParallelInput,
                                            )
                                        ]),
            T5Attention:
                ModulePolicyDescription(attribute_replacement={
                    "d_model":
                        self.model.config.d_model // self.shard_config.tensor_parallel_size,
                    "n_heads":
                        self.model.config.num_heads // self.shard_config.tensor_parallel_size,
                    "inner_dim":
                        self.model.config.num_heads * self.model.config.d_kv // self.shard_config.tensor_parallel_size
                },
                                        param_replacement=[],
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
                                            SubModuleReplacementDescription(suffix="relative_attention_bias",
                                                                            target_module=Embedding1D,
                                                                            kwargs=dict(gather_output=False),
                                                                            ignore_if_not_exist=True)
                                        ]),
            T5LayerFF:
                ModulePolicyDescription(attribute_replacement={},
                                        param_replacement=[],
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(
                                                suffix="dropout",
                                                target_module=DropoutForParallelInput,
                                            ),
                                        ]),
            T5DenseGatedActDense:
                ModulePolicyDescription(attribute_replacement={},
                                        param_replacement=[],
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(
                                                suffix="wi_0",
                                                target_module=Linear1D_Col,
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="wi_1",
                                                target_module=Linear1D_Row,
                                            ),
                                            SubModuleReplacementDescription(suffix="wo",
                                                                            target_module=Linear1D_Col,
                                                                            kwargs=dict(gather_output=True)),
                                            SubModuleReplacementDescription(
                                                suffix="dropout",
                                                target_module=DropoutForParallelInput,
                                            )
                                        ]),
            T5DenseActDense:
                ModulePolicyDescription(attribute_replacement={},
                                        param_replacement=[],
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
                                            )
                                        ])
        }

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            base_policy[T5LayerFF].sub_module_replacement.append(
                SubModuleReplacementDescription(suffix="layer_norm", target_module=FusedRMSNorm))
            base_policy[T5LayerSelfAttention].sub_module_replacement.append(
                SubModuleReplacementDescription(suffix="layer_norm", target_module=FusedRMSNorm))
            base_policy[T5LayerCrossAttention].sub_module_replacement.append(
                SubModuleReplacementDescription(suffix="layer_norm", target_module=FusedRMSNorm))
            base_policy[T5Stack].sub_module_replacement.append(
                SubModuleReplacementDescription(suffix="final_layer_norm", target_module=FusedRMSNorm))

        return base_policy

    def new_model_class(self):
        return None

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
        base_policy[T5Model] = ModulePolicyDescription(attribute_replacement={},
                                                       param_replacement=[],
                                                       sub_module_replacement=[
                                                           SubModuleReplacementDescription(
                                                               suffix="shared",
                                                               target_module=VocabParallelEmbedding1D,
                                                           )
                                                       ])
        return base_policy


class T5ForConditionalGenerationPolicy(T5BasePolicy):

    def module_policy(self):
        from transformers import T5ForConditionalGeneration

        policy = super().module_policy()
        policy[T5ForConditionalGeneration] = ModulePolicyDescription(attribute_replacement={},
                                                                     param_replacement=[],
                                                                     sub_module_replacement=[
                                                                         SubModuleReplacementDescription(
                                                                             suffix="shared",
                                                                             target_module=VocabParallelEmbedding1D,
                                                                         ),
                                                                         SubModuleReplacementDescription(
                                                                             suffix="lm_head",
                                                                             target_module=Linear1D_Col,
                                                                             kwargs=dict(gather_output=True))
                                                                     ])
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
        base_policy[T5EncoderModel] = ModulePolicyDescription(attribute_replacement={},
                                                              param_replacement=[],
                                                              sub_module_replacement=[
                                                                  SubModuleReplacementDescription(
                                                                      suffix="shared",
                                                                      target_module=VocabParallelEmbedding1D,
                                                                  )
                                                              ])
        return base_policy

    def postprocess(self):
        binding_map = [
            ["shared", "encoder.embed_tokens"],
        ]

        for k, v in binding_map:
            mod = getattr_(self.model, k)
            setattr_(self.model, v, mod)
        return self.model

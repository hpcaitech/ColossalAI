from colossalai.shardformer.layer import FusedLayerNorm, Linear1D_Col, Linear1D_Row, VocabParallelEmbedding1D

from .._utils import getattr_, setattr_
from .basepolicy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = [
    'OPTPolicy', 'OPTModelPolicy', 'OPTForCausalLMPolicy', 'OPTForSequenceClassificationPolicy',
    'OPTForQuestionAnsweringPolicy'
]


class OPTPolicy(Policy):

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
        from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoder, OPTDecoderLayer

        policy = {}

        if self.shard_config.enable_tensor_parallelism:
            policy[OPTDecoder] = ModulePolicyDescription(sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="embed_tokens",
                    target_module=VocabParallelEmbedding1D,
                )
            ])
            policy[OPTDecoderLayer] = ModulePolicyDescription(sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="fc1",
                    target_module=Linear1D_Col,
                ),
                SubModuleReplacementDescription(
                    suffix="fc2",
                    target_module=Linear1D_Row,
                )
            ])

            policy[OPTAttention] = ModulePolicyDescription(attribute_replacement={
                "embed_dim": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "num_heads": self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size
            },
                                                           sub_module_replacement=[
                                                               SubModuleReplacementDescription(
                                                                   suffix="q_proj",
                                                                   target_module=Linear1D_Col,
                                                               ),
                                                               SubModuleReplacementDescription(
                                                                   suffix="k_proj",
                                                                   target_module=Linear1D_Col,
                                                               ),
                                                               SubModuleReplacementDescription(
                                                                   suffix="v_proj",
                                                                   target_module=Linear1D_Col,
                                                               ),
                                                               SubModuleReplacementDescription(
                                                                   suffix="out_proj",
                                                                   target_module=Linear1D_Row,
                                                               ),
                                                           ])

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="final_layer_norm", target_module=FusedLayerNorm, ignore_if_not_exist=True),
                                                        policy=policy,
                                                        target_key=OPTDecoder)
            self.append_or_create_submodule_replacement(description=[
                SubModuleReplacementDescription(suffix="self_attn_layer_norm",
                                                target_module=FusedLayerNorm,
                                                ignore_if_not_exist=True),
                SubModuleReplacementDescription(suffix="final_layer_norm",
                                                target_module=FusedLayerNorm,
                                                ignore_if_not_exist=True)
            ],
                                                        policy=policy,
                                                        target_key=OPTDecoderLayer)

        return policy

    def postprocess(self):
        return self.model


class OPTModelPolicy(OPTPolicy):

    def __init__(self) -> None:
        super().__init__()


class OPTForCausalLMPolicy(OPTPolicy):

    def module_policy(self):
        from transformers.models.opt.modeling_opt import OPTForCausalLM

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="lm_head", target_module=Linear1D_Col, kwargs=dict(gather_output=True)),
                                                        policy=policy,
                                                        target_key=OPTForCausalLM)
        return policy

    def postprocess(self):
        binding_map = {
            'model.decoder.embed_tokens': 'lm_head',
        }

        for k, v in binding_map.items():
            src_mod = getattr_(self.model, k)
            dst_mod = getattr_(self.model, v)
            dst_mod.weight = src_mod.weight

        return self.model


class OPTForSequenceClassificationPolicy(OPTPolicy):

    def __init__(self) -> None:
        super().__init__()


class OPTForQuestionAnsweringPolicy(OPTPolicy):

    def __init__(self) -> None:
        super().__init__()

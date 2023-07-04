import torch.nn as nn

import colossalai.shardformer.layer as col_nn

from .._utils import getattr_, setattr_
from .basepolicy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

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


# GPT2Model
class GPT2ModelPolicy(GPT2Policy):

    def __init__(self) -> None:
        super().__init__()


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
        return module_policy

    def postprocess(self):
        binding_map = {"transformer.wte.weight": "lm_head.weight"}
        for k, v in binding_map.items():
            param = getattr_(self.model, k)
            setattr_(self.model, v, param)
        return self.model


# GPT22DoubleHeadsModel
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
        return module_policy

    def postprocess(self):
        binding_map = {"transformer.wte.weight": "lm_head.weight"}
        for k, v in binding_map.items():
            param = getattr_(self.model, k)
            setattr_(self.model, v, param)
        return self.model


# GPT2ForTokenClassification
class GPT2ForTokenClassificationPolicy(GPT2Policy):

    def __init__(self) -> None:
        super().__init__()


# GPT2ForSequenceClassification
class GPT2ForSequenceClassificationPolicy(GPT2Policy):

    def __init__(self) -> None:
        super().__init__()

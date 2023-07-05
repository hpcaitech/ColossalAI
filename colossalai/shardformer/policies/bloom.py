import torch.nn as nn

import colossalai.shardformer.layer as col_nn

from .._utils import getattr_, setattr_
from ..modeling.bloom import build_bloom_alibi_tensor_fn
from .basepolicy import ModulePolicyDescription, Policy, SubModuleReplacementDescription


class BloomPolicy(Policy):

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
    pass


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
        binding_map = {"transformer.word_embeddings.weight": "lm_head.weight"}

        for k, v in binding_map.items():
            param = getattr_(self.model, k)

            if not isinstance(param, nn.Parameter):
                param = nn.Parameter(param)

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

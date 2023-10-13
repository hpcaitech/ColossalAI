import warnings
from functools import partial
from typing import Callable, Dict, List

from torch import Tensor, nn

import colossalai.shardformer.layer as col_nn

from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription
from ..modeling.falcon import (
    build_falcon_alibi_tensor_fn, 
    get_tp_falcon_decoder_layer_forward,
    get_falcon_flash_attention_forward
)
__all__ = [
    "FalconPolicy"
]

class FalconPolicy(Policy):
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
        from transformers.models.falcon.modeling_falcon import FalconModel, FalconDecoderLayer, FalconAttention

        if not self.model.config.new_decoder_architecture and self.model.config.multi_query:
            warnings.warn("Falcon dosen't support tensor parallelism when (not new_decoder_architecture and multi_query) is True, will ignore the tensor parallelism flag.")
            self.shard_config.enable_tensor_parallelism = False

        policy = {}
        if self.shard_config.enable_tensor_parallelism:
            attn_attribute_replacement={
                "self_attention.hidden_size": self.model.config.hidden_size
                // self.shard_config.tensor_parallel_size,
                "self_attention.split_size": self.model.config.hidden_size
                // self.shard_config.tensor_parallel_size,
                "self_attention.num_heads": self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
                "self_attention.num_kv_heads": self.model.config.num_kv_heads // self.shard_config.tensor_parallel_size,
            }
            
            policy[FalconDecoderLayer] = ModulePolicyDescription(
                attribute_replacement=attn_attribute_replacement,
                method_replacement={
                    "forward": get_tp_falcon_decoder_layer_forward()
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
                        target_module=col_nn.Linear1D_Row
                    ),
                ]
            )
            

            policy[FalconModel] = ModulePolicyDescription(
                attribute_replacement={
                    "num_heads": self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
                },
                method_replacement={
                    "build_alibi_tensor": build_falcon_alibi_tensor_fn(self.shard_config.tensor_parallel_process_group)
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="word_embeddings",
                        target_module=col_nn.VocabParallelEmbedding1D,
                    )
                ],
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
                        suffix="ln_attn",
                        target_module=col_nn.FusedLayerNorm,
                        ignore_if_not_exist=True
                    ),
                    SubModuleReplacementDescription(
                        suffix="ln_mlp",
                        target_module=col_nn.FusedLayerNorm,
                        ignore_if_not_exist=True
                    ),
                    SubModuleReplacementDescription(
                        suffix="input_layernorm",
                        target_module=col_nn.FusedLayerNorm,
                        ignore_if_not_exist=True
                    ),
                    SubModuleReplacementDescription(
                        suffix="post_attention_layernorm",
                        target_module=col_nn.FusedLayerNorm,
                        ignore_if_not_exist=True
                    ),
                ],
                policy=policy,
                target_key=FalconDecoderLayer,
            )

        if self.shard_config.enable_flash_attention:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_falcon_flash_attention_forward()
                },
                policy=policy,
                target_key=FalconAttention,
            )
        print(policy)
        return policy

    def postprocess(self):
        return self.model

class FalconModelPolicy(FalconPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()
        return policy
    
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
                    suffix="lm_head", target_module=col_nn.Linear1D_Col, kwargs=dict(gather_output=True)
                ),
                policy=policy,
                target_key=FalconForCausalLM,
            )
        return policy
    
class FalconForSequenceClassificationPolicy(FalconPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.falcon.modeling_falcon import FalconForSequenceClassification

        policy = super().module_policy()

        # handle tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="score", target_module=col_nn.Linear1D_Col, kwargs=dict(gather_output=True)
                ),
                policy=policy,
                target_key=FalconForSequenceClassification,
            )
        return policy
    
class FalconForTokenClassificationPolicy(FalconPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.falcon.modeling_falcon import FalconForTokenClassification

        policy = super().module_policy()

        # handle tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                    suffix="classifier", target_module=col_nn.Linear1D_Col, kwargs=dict(gather_output=True)
                    ),
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=col_nn.DropoutForReplicatedInput,
                    )
                ],
                policy=policy,
                target_key=FalconForTokenClassification,
            )
        return policy
    
class FalconForQuestionAnsweringPolicy(FalconPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.falcon.modeling_falcon import FalconForQuestionAnswering

        policy = super().module_policy()

        # handle tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="qa_outputs", target_module=col_nn.Linear1D_Col, kwargs=dict(gather_output=True)
                ),
                policy=policy,
                target_key=FalconForQuestionAnswering,
            )
        return policy
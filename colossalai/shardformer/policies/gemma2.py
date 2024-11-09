from functools import partial
from typing import Callable, Dict, List, Union

import torch.nn as nn
from torch import Tensor
from colossalai.shardformer.layer import (
    Linear1D_Col,
    Linear1D_Row,
    PaddingEmbedding,
    PaddingLMHead,
    RMSNorm,
    VocabParallelEmbedding1D,
    VocabParallelLMHead1D,
)

from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription
from ..modeling.gemma2 import Gemma2PipelineForwards
__all__ = ["Gemma2Policy", "Gemma2ForCausalLMPolicy"]

class Gemma2Policy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        self.tie_weight = self.tie_weight_check()
        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        from transformers.models.gemma2.modeling_gemma2 import (
            Gemma2DecoderLayer,
            Gemma2Model,
        )
        policy = {}

        embedding_cls = None
        if self.shard_config.enable_tensor_parallelism:
            embedding_cls = VocabParallelEmbedding1D
        else:
            if self.tie_weight:
                embedding_cls = PaddingEmbedding

        norm_cls = RMSNorm

        if self.shard_config.enable_tensor_parallelism:
            tp_size = self.shard_config.tensor_parallel_size
            num_q_heads = self.model.config.num_attention_heads // tp_size
            decoder_attribute_replacement = {
                "self_attn.hidden_size": self.model.config.hidden_size // tp_size,
                "self_attn.num_heads": num_q_heads,
            }
            num_kv_heads = self.model.config.num_key_value_heads // tp_size
            decoder_attribute_replacement["self_attn.num_key_value_heads"] = num_kv_heads
            policy[Gemma2DecoderLayer] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="mlp.gate_proj", 
                        target_module=Linear1D_Col),
                    SubModuleReplacementDescription(
                        suffix="mlp.up_proj", 
                        target_module=Linear1D_Col),
                    SubModuleReplacementDescription(
                        suffix="mlp.down_proj", 
                        target_module=Linear1D_Row),
                    SubModuleReplacementDescription(
                        suffix="self_attn.q_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.k_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.v_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=Linear1D_Row,
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
                target_key=Gemma2Model,
            )

        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="input_layernorm", 
                    target_module=norm_cls),
                SubModuleReplacementDescription(
                    suffix="pre_feedforward_layernorm", 
                    target_module=norm_cls),
                SubModuleReplacementDescription(
                    suffix="post_feedforward_layernorm", 
                    target_module=norm_cls),
                SubModuleReplacementDescription(
                    suffix="post_attention_layernorm", 
                    target_module=norm_cls),
            ],
            policy=policy,
            target_key=Gemma2DecoderLayer,
        )

        self.append_or_create_submodule_replacement(
            description=SubModuleReplacementDescription(
                suffix="norm",
                target_module=norm_cls,
            ),
            policy=policy,
            target_key=Gemma2Model,
        )
        return policy

    def postprocess(self):
        return self.model


class Gemma2ForCausalLMPolicy(Gemma2Policy):
    def module_policy(self):
        from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="lm_head",
                    target_module=VocabParallelLMHead1D,
                    kwargs=dict(
                        gather_output=not self.shard_config.parallel_output,
                        make_vocab_size_divisible_by=self.shard_config.make_vocab_size_divisible_by,
                    ),
                ),
                policy=policy,
                target_key=Gemma2ForCausalLM,
            )
            if self.shard_config.parallel_output:
                method_replacement = {"forward": partial(Gemma2PipelineForwards.gemma2_for_causal_lm_forward, shard_config=self.shard_config)}
                self.append_or_create_method_replacement(
                    description=method_replacement, policy=policy, target_key=Gemma2ForCausalLM
                )
        else:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="lm_head",
                    target_module=PaddingLMHead,
                    kwargs=dict(make_vocab_size_divisible_by=self.shard_config.make_vocab_size_divisible_by),
                ),
                policy=policy,
                target_key=Gemma2ForCausalLM,
            )

        return policy

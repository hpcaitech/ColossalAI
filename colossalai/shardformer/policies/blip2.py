import colossalai.shardformer.layer as col_nn

from ..modeling.blip2 import (
    forward_fn,
    get_blip2_flash_attention_forward,
    get_jit_fused_blip2_mlp_forward,
    get_jit_fused_blip2_QFormer_output_forward,
    get_jit_fused_blip2_QFormer_self_output_forward,
)
from ..modeling.jit import get_jit_fused_dropout_add_func
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["BlipPolicy", "BlipModelPolicy"]


class BlipPolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        self.tie_weight = self.tie_weight_check()
        self.enable_bias_gelu_fused = (
            self.shard_config.enable_jit_fused and self.model.config.vision_config.hidden_act == "gelu"
        )
        return self.model

    def module_policy(self):
        from transformers.models.blip_2.modeling_blip_2 import (
            Blip2Attention,
            Blip2EncoderLayer,
            Blip2MLP,
            Blip2QFormerLayer,
            Blip2QFormerModel,
            Blip2QFormerOutput,
            Blip2QFormerSelfOutput,
            Blip2VisionModel,
        )
        from transformers.models.opt.modeling_opt import OPTDecoderLayer, OPTForCausalLM

        policy = {}

        embedding_cls = None
        if self.shard_config.enable_tensor_parallelism:
            embedding_cls = col_nn.VocabParallelEmbedding1D
        else:
            if self.tie_weight:
                embedding_cls = col_nn.PaddingEmbedding

        if self.shard_config.enable_fused_normalization:
            norm_cls = col_nn.FusedLayerNorm
        else:
            norm_cls = col_nn.LayerNorm

        if self.shard_config.enable_tensor_parallelism:
            assert (
                self.model.config.vision_config.num_attention_heads % self.shard_config.tensor_parallel_size == 0
            ), f"The number of attention heads must be divisible by tensor parallel size."
            policy[Blip2EncoderLayer] = ModulePolicyDescription(
                attribute_replacement={
                    "self_attn.num_heads": self.model.config.vision_config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                    "self_attn.embed_dim": self.model.config.vision_config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.qkv",
                        target_module=col_nn.FusedLinear1D_Col,
                        kwargs={
                            "n_fused": 3,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.projection",
                        target_module=col_nn.Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.fc1",
                        target_module=col_nn.Linear1D_Col,
                        kwargs={"skip_bias_add": self.enable_bias_gelu_fused},
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.fc2",
                        target_module=col_nn.Linear1D_Row,
                    ),
                ],
            )

            policy[Blip2QFormerModel] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                ]
            )

            policy[Blip2QFormerLayer] = ModulePolicyDescription(
                attribute_replacement={
                    "attention.attention.num_attention_heads": self.model.config.qformer_config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                    "attention.attention.all_head_size": self.model.config.qformer_config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                    "crossattention.attention.num_attention_heads": self.model.config.qformer_config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                    "crossattention.attention.all_head_size": self.model.config.qformer_config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="attention.attention.query",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.attention.key",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.attention.value",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.attention.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.output.dense",
                        target_module=col_nn.Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.output.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="crossattention.attention.query",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="crossattention.attention.key",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="crossattention.attention.value",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="crossattention.attention.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="crossattention.output.dense",
                        target_module=col_nn.Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="crossattention.output.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="intermediate_query.dense",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="output_query.dense",
                        target_module=col_nn.Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="output_query.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                ],
            )

            policy[OPTDecoderLayer] = ModulePolicyDescription(
                attribute_replacement={
                    "self_attn.embed_dim": self.model.config.text_config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                    "self_attn.num_heads": self.model.config.text_config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.q_proj",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.k_proj",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.v_proj",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.out_proj",
                        target_module=col_nn.Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="fc1",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="fc2",
                        target_module=col_nn.Linear1D_Row,
                    ),
                ],
            )

            policy[Blip2Attention] = ModulePolicyDescription(method_replacement={"forward": forward_fn()})
            if self.enable_bias_gelu_fused:
                self.append_or_create_method_replacement(
                    description={
                        "forward": get_jit_fused_blip2_mlp_forward(),
                    },
                    policy=policy,
                    target_key=Blip2MLP,
                )

        if embedding_cls is not None:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="model.decoder.embed_tokens",
                        target_module=embedding_cls,
                        kwargs={"make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by},
                    ),
                ],
                policy=policy,
                target_key=OPTForCausalLM,
            )

        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="lm_head",
                        target_module=col_nn.VocabParallelLMHead1D,
                        kwargs={
                            "gather_output": True,
                            "make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by,
                        },
                    ),
                ],
                policy=policy,
                target_key=OPTForCausalLM,
            )
        else:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="lm_head",
                        target_module=col_nn.PaddingLMHead,
                        kwargs={"make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by},
                    ),
                ],
                policy=policy,
                target_key=OPTForCausalLM,
            )
        # optimization configuration
        # Handle Blip2EncoderLayer layer
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="layer_norm1",
                    target_module=norm_cls,
                ),
                SubModuleReplacementDescription(
                    suffix="layer_norm2",
                    target_module=norm_cls,
                ),
            ],
            policy=policy,
            target_key=Blip2EncoderLayer,
        )

        # handle Blip2VisionModel layer
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="post_layernorm",
                    target_module=norm_cls,
                )
            ],
            policy=policy,
            target_key=Blip2VisionModel,
        )

        # handle Blip2VisionModel layer
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="layernorm",
                    target_module=norm_cls,
                )
            ],
            policy=policy,
            target_key=Blip2QFormerModel,
        )

        # handle Blip2QFormerLayer layer
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="attention.output.LayerNorm",
                    target_module=norm_cls,
                ),
                SubModuleReplacementDescription(
                    suffix="crossattention.output.LayerNorm",
                    target_module=norm_cls,
                ),
                SubModuleReplacementDescription(
                    suffix="output_query.LayerNorm",
                    target_module=norm_cls,
                ),
            ],
            policy=policy,
            target_key=Blip2QFormerLayer,
        )

        # handle OPTForCausalLM layer
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="model.decoder.final_layer_norm",
                    target_module=norm_cls,
                )
            ],
            policy=policy,
            target_key=OPTForCausalLM,
        )

        # handle OPTDecoderLayer layer
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="self_attn_layer_norm",
                    target_module=norm_cls,
                ),
                SubModuleReplacementDescription(
                    suffix="final_layer_norm",
                    target_module=norm_cls,
                ),
            ],
            policy=policy,
            target_key=OPTDecoderLayer,
        )

        # use flash attention
        if self.shard_config.enable_flash_attention:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_blip2_flash_attention_forward(),
                },
                policy=policy,
                target_key=Blip2Attention,
            )

        # use jit operator
        if self.shard_config.enable_jit_fused:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_blip2_QFormer_self_output_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=Blip2QFormerSelfOutput,
            )
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_blip2_QFormer_output_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=Blip2QFormerOutput,
            )

        return policy

    def postprocess(self):
        return self.model


# Blip2Model
class Blip2ModelPolicy(BlipPolicy):
    def __init__(self) -> None:
        super().__init__()


# Blip2ForConditionalGeneration
class Blip2ForConditionalGenerationPolicy(BlipPolicy):
    def __init__(self) -> None:
        super().__init__()

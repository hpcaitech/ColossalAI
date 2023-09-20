import colossalai.shardformer.layer as col_nn

from ..modeling.sam import forward_fn, get_sam_flash_attention_forward, get_sam_vision_flash_attention_forward
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["SamPolicy", "SamModelPolicy"]


class SamPolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        return self.model

    def module_policy(self):
        from transformers.models.sam.modeling_sam import (
            SamAttention,
            SamTwoWayAttentionBlock,
            SamTwoWayTransformer,
            SamVisionAttention,
            SamVisionLayer,
        )

        policy = {}

        if self.shard_config.enable_tensor_parallelism:
            policy[SamVisionLayer] = ModulePolicyDescription(
                attribute_replacement={
                    "attn.num_attention_heads": self.model.config.vision_config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="attn.qkv",
                        target_module=col_nn.FusedLinear1D_Col,
                        kwargs={
                            "n_fused": 3,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="attn.proj",
                        target_module=col_nn.Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.lin1",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.lin2",
                        target_module=col_nn.Linear1D_Row,
                    ),
                ],
            )
            policy[SamTwoWayAttentionBlock] = ModulePolicyDescription(
                attribute_replacement={
                    "self_attn.num_attention_heads": self.model.config.mask_decoder_config.num_attention_heads
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
                        suffix="cross_attn_token_to_image.q_proj",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="cross_attn_token_to_image.k_proj",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="cross_attn_token_to_image.v_proj",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="cross_attn_token_to_image.out_proj",
                        target_module=col_nn.Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.lin1",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.lin2",
                        target_module=col_nn.Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="cross_attn_image_to_token.q_proj",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="cross_attn_image_to_token.k_proj",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="cross_attn_image_to_token.v_proj",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="cross_attn_image_to_token.out_proj",
                        target_module=col_nn.Linear1D_Row,
                    ),
                ],
            )
            policy[SamTwoWayTransformer] = ModulePolicyDescription(
                attribute_replacement={
                    "final_attn_token_to_image.num_attention_heads": self.model.config.mask_decoder_config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="final_attn_token_to_image.q_proj",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="final_attn_token_to_image.k_proj",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="final_attn_token_to_image.v_proj",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="final_attn_token_to_image.out_proj",
                        target_module=col_nn.Linear1D_Row,
                    ),
                ],
            )

            # add `DropoutForParallelInput` layer to replace the useage of `nn.functional.dropout`
            policy[SamVisionAttention] = ModulePolicyDescription(
                attribute_replacement={
                    "dropout_layer": col_nn.DropoutForParallelInput(self.model.config.vision_config.attention_dropout)
                },
                method_replacement={"forward": forward_fn()},
                sub_module_replacement=[],
            )

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            # Handle SamVisionLayer
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="layer_norm1",
                        target_module=col_nn.FusedLayerNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="layer_norm2",
                        target_module=col_nn.FusedLayerNorm,
                    ),
                ],
                policy=policy,
                target_key=SamVisionLayer,
            )

            # Handle SamTwoWayAttentionBlock
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="layer_norm1",
                        target_module=col_nn.FusedLayerNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="layer_norm2",
                        target_module=col_nn.FusedLayerNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="layer_norm3",
                        target_module=col_nn.FusedLayerNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="layer_norm4",
                        target_module=col_nn.FusedLayerNorm,
                    ),
                ],
                policy=policy,
                target_key=SamTwoWayAttentionBlock,
            )

            # Handle SamTwoWayTransformer
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="layer_norm_final_attn",
                        target_module=col_nn.FusedLayerNorm,
                    )
                ],
                policy=policy,
                target_key=SamTwoWayTransformer,
            )

        # use flash attention
        if self.shard_config.enable_flash_attention:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_sam_flash_attention_forward(),
                },
                policy=policy,
                target_key=SamAttention,
            )
            self.append_or_create_method_replacement(
                description={
                    "forward": get_sam_vision_flash_attention_forward(),
                },
                policy=policy,
                target_key=SamVisionAttention,
            )

        return policy

    def postprocess(self):
        return self.model


# SamModel
class SamModelPolicy(SamPolicy):
    def __init__(self) -> None:
        super().__init__()

from diffusers.models.attention import JointTransformerBlock
from diffusers.models.transformers import SD3Transformer2DModel
from torch import nn

from colossalai.inference.config import RPC_PARAM
from colossalai.inference.modeling.layers.diffusion import DiffusionPipe
from colossalai.inference.modeling.layers.distrifusion import (
    DistrifusionConv2D,
    DistrifusionFusedAttention,
    DistrifusionPatchEmbed,
    SD3Transformer2DModel_forward,
)
from colossalai.inference.modeling.models.stablediffusion3 import sd3_forward
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription


class StableDiffusion3InferPolicy(Policy, RPC_PARAM):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = {}

        if self.shard_config.extra_kwargs["model_shard_infer_config"].patched_parallelism_size > 1:

            policy[SD3Transformer2DModel] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="pos_embed.proj",
                        target_module=DistrifusionConv2D,
                        kwargs={"model_shard_infer_config": self.shard_config.extra_kwargs["model_shard_infer_config"]},
                    ),
                    SubModuleReplacementDescription(
                        suffix="pos_embed",
                        target_module=DistrifusionPatchEmbed,
                        kwargs={"model_shard_infer_config": self.shard_config.extra_kwargs["model_shard_infer_config"]},
                    ),
                ],
                attribute_replacement={
                    "patched_parallel_size": self.shard_config.extra_kwargs[
                        "model_shard_infer_config"
                    ].patched_parallelism_size
                },
                method_replacement={"forward": SD3Transformer2DModel_forward},
            )

        policy[JointTransformerBlock] = ModulePolicyDescription(
            sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="attn",
                    target_module=DistrifusionFusedAttention,
                    kwargs={
                        "model_shard_infer_config": self.shard_config.extra_kwargs["model_shard_infer_config"],
                    },
                )
            ]
        )

        self.append_or_create_method_replacement(
            description={"forward": sd3_forward}, policy=policy, target_key=DiffusionPipe
        )
        return policy

    def preprocess(self) -> nn.Module:
        return self.model

    def postprocess(self):
        return self.model

    def config_sanity_check(self):
        pass

    def to_rpc_param(self) -> str:
        return __class__.__name__

    @staticmethod
    def from_rpc_param() -> "StableDiffusion3InferPolicy":
        return StableDiffusion3InferPolicy()

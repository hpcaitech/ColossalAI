from typing import List

from torch.nn import Module

from colossalai.shardformer.layer import Linear1D_Col
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, SubModuleReplacementDescription
from colossalai.shardformer.policies.llama import LlamaPolicy

from ..modeling.llama import LlamaPipelineForwards


class LlamaForCausalLMPipelinePolicy(LlamaPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers import LlamaForCausalLM

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            # add a new item for casual lm
            new_item = {
                LlamaForCausalLM: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head", target_module=Linear1D_Col, kwargs=dict(gather_output=True)
                        )
                    ]
                )
            }
            policy.update(new_item)

        if self.pipeline_stage_manager:
            # set None as default
            self.set_pipeline_forward(
                model_cls=LlamaForCausalLM, new_forward=LlamaPipelineForwards.llama_for_causal_lm_forward, policy=policy
            )

        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_first_stage():
            held_layers.append(self.model.lm_head)
        return held_layers

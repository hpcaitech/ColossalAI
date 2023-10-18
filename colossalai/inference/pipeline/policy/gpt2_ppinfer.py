from functools import partial
from typing import Callable, Dict, List

from torch import Tensor, nn

import colossalai.shardformer.layer as col_nn
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription
from colossalai.shardformer.policies.gpt2 import GPT2Policy

from ..modeling.gpt2 import GPT2PipelineForwards


class GPT2LMHeadModelPipelinePolicy(GPT2Policy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

        module_policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            addon_module = {
                GPT2LMHeadModel: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head", target_module=col_nn.Linear1D_Col, kwargs={"gather_output": True}
                        )
                    ]
                )
            }
            module_policy.update(addon_module)

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(
                model_cls=GPT2LMHeadModel,
                new_forward=GPT2PipelineForwards.gpt2_lmhead_model_forward,
                policy=module_policy,
            )
        return module_policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        # make the tie weight lm_head and embedding in the same device to save memory
        # if self.pipeline_stage_manager.is_first_stage():
        if self.pipeline_stage_manager.is_first_stage():
            held_layers.append(self.model.lm_head)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """The weights of wte and lm_head are shared."""
        module = self.model
        stage_manager = self.pipeline_stage_manager
        if stage_manager is not None:
            if stage_manager.num_stages > 1 and id(module.transformer.wte.weight) == id(module.lm_head.weight):
                first_stage, last_stage = 0, stage_manager.num_stages - 1
                return [{first_stage: module.transformer.wte.weight, last_stage: module.lm_head.weight}]
        return []

    def set_pipeline_forward(self, model_cls: nn.Module, new_forward: Callable, policy: Dict) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
        to customized forward method, and add this changing to policy."""
        if not self.pipeline_stage_manager:
            raise ValueError("set_pipeline_forward method can only be called when pipeline parallel is enabled.")
        stage_manager = self.pipeline_stage_manager
        if self.model.__class__.__name__ == "GPT2Model":
            module = self.model
        else:
            module = self.model.transformer

        layers_per_stage = Policy.distribute_layers(len(module.h), stage_manager.num_stages)
        stage_index = Policy.get_stage_index(layers_per_stage, stage_manager.stage)
        method_replacement = {"forward": partial(new_forward, stage_manager=stage_manager, stage_index=stage_index)}
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=model_cls)

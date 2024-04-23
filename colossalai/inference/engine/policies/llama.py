from functools import partial
from typing import List

import torch
from torch.nn import Module
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
)

from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, SubModuleReplacementDescription

# import colossalai
from colossalai.shardformer.policies.llama import LlamaForCausalLMPolicy

from ..modeling._utils import init_to_get_rotary
from ..modeling.llama import LlamaInferenceForwards

try:
    from colossalai.kernel.triton import rmsnorm_forward

    HAS_TRITON_RMSNORM = True
except:
    print("you should install triton from https://github.com/openai/triton")
    HAS_TRITON_RMSNORM = False


def get_triton_rmsnorm_forward():
    if HAS_TRITON_RMSNORM:

        def _triton_rmsnorm_forward(self: LlamaRMSNorm, hidden_states: torch.Tensor):
            return rmsnorm_forward(hidden_states, self.weight.data, self.variance_epsilon)

        return _triton_rmsnorm_forward
    else:
        return None


class LlamaModelInferPolicy(LlamaForCausalLMPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()
        decoder_attribute_replacement = {
            "self_attn.hidden_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
            "self_attn.num_heads": self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
            "self_attn.num_key_value_heads": self.model.config.num_key_value_heads
            // self.shard_config.tensor_parallel_size,
        }
        if self.shard_config.extra_kwargs.get("quant", None) == "gptq":
            from colossalai.inference.quant.gptq.cai_gptq import ColCaiQuantLinear, RowCaiQuantLinear

            policy[LlamaDecoderLayer] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.q_proj",
                        target_module=ColCaiQuantLinear,
                        kwargs={"split_num": 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.k_proj",
                        target_module=ColCaiQuantLinear,
                        kwargs={"split_num": 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.v_proj",
                        target_module=ColCaiQuantLinear,
                        kwargs={"split_num": 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=RowCaiQuantLinear,
                        kwargs={"split_num": 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.gate_proj",
                        target_module=ColCaiQuantLinear,
                        kwargs={"split_num": 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.up_proj",
                        target_module=ColCaiQuantLinear,
                        kwargs={"split_num": 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.down_proj",
                        target_module=RowCaiQuantLinear,
                        kwargs={"split_num": 1},
                    ),
                ],
            )

        elif self.shard_config.extra_kwargs.get("quant", None) == "smoothquant":
            from colossalai.inference.quant.smoothquant.models.llama import LlamaSmoothquantDecoderLayer
            from colossalai.inference.quant.smoothquant.models.parallel_linear import (
                ColW8A8BFP32OFP32Linear,
                RowW8A8B8O8Linear,
                RowW8A8BFP32O32LinearSiLU,
                RowW8A8BFP32OFP32Linear,
            )

            policy[LlamaSmoothquantDecoderLayer] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.q_proj",
                        target_module=RowW8A8B8O8Linear,
                        kwargs={"split_num": 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.k_proj",
                        target_module=RowW8A8B8O8Linear,
                        kwargs={"split_num": 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.v_proj",
                        target_module=RowW8A8B8O8Linear,
                        kwargs={"split_num": 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=ColW8A8BFP32OFP32Linear,
                        kwargs={"split_num": 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.gate_proj",
                        target_module=RowW8A8BFP32O32LinearSiLU,
                        kwargs={"split_num": 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.up_proj",
                        target_module=RowW8A8BFP32OFP32Linear,
                        kwargs={"split_num": 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.down_proj",
                        target_module=ColW8A8BFP32OFP32Linear,
                        kwargs={"split_num": 1},
                    ),
                ],
            )
        self.shard_config._infer()

        infer_forward = LlamaInferenceForwards.llama_model_forward
        method_replacement = {"forward": partial(infer_forward)}
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=LlamaModel)

        infer_forward = LlamaInferenceForwards.llama_decoder_layer_forward
        method_replacement = {"forward": partial(infer_forward)}
        self.append_or_create_method_replacement(
            description=method_replacement, policy=policy, target_key=LlamaDecoderLayer
        )

        infer_forward = LlamaInferenceForwards.llama_flash_attn_kvcache_forward
        method_replacement = {"forward": partial(infer_forward)}
        self.append_or_create_method_replacement(
            description=method_replacement, policy=policy, target_key=LlamaAttention
        )

        # set as default, in inference we also use pipeline style forward, just setting stage as 1
        self.set_pipeline_forward(
            model_cls=LlamaForCausalLM, new_forward=LlamaInferenceForwards.llama_causal_lm_forward, policy=policy
        )

        infer_forward = None
        if HAS_TRITON_RMSNORM:
            infer_forward = get_triton_rmsnorm_forward()

        if infer_forward is not None:
            method_replacement = {"forward": partial(infer_forward)}
            self.append_or_create_method_replacement(
                description=method_replacement, policy=policy, target_key=LlamaRMSNorm
            )

        return policy

    def postprocess(self):
        init_to_get_rotary(self.model.model)
        return self.model

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None

        if self.model.__class__.__name__ == "LlamaModel":
            module = self.model
        else:
            module = self.model.model
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = stage_manager.distribute_layers(len(module.layers))
        if stage_manager.is_first_stage():
            held_layers.append(module.embed_tokens)
            held_layers.append(self.model.lm_head)
        start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        held_layers.extend(module.layers[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.norm)

        return held_layers

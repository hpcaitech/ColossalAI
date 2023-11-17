from functools import partial
from typing import List

import torch
from torch.nn import LayerNorm, Module

import colossalai.shardformer.layer as col_nn
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, SubModuleReplacementDescription
from colossalai.shardformer.policies.bloom import BloomForCausalLMPolicy

from ..modeling.bloom import BloomInferenceForwards

try:
    from colossalai.kernel.triton import layer_norm

    HAS_TRITON_NORM = True
except:
    print("Some of our kernels require triton. You might want to install triton from https://github.com/openai/triton")
    HAS_TRITON_NORM = False


def get_triton_layernorm_forward():
    if HAS_TRITON_NORM:

        def _triton_layernorm_forward(self: LayerNorm, hidden_states: torch.Tensor):
            return layer_norm(hidden_states, self.weight.data, self.bias, self.eps)

        return _triton_layernorm_forward
    else:
        return None


class BloomModelInferPolicy(BloomForCausalLMPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.bloom.modeling_bloom import BloomAttention, BloomBlock, BloomForCausalLM, BloomModel

        policy = super().module_policy()
        if self.shard_config.extra_kwargs.get("quant", None) == "gptq":
            from colossalai.inference.quant.gptq.cai_gptq import ColCaiQuantLinear, RowCaiQuantLinear

            policy[BloomBlock] = ModulePolicyDescription(
                attribute_replacement={
                    "self_attention.hidden_size": self.model.config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                    "self_attention.split_size": self.model.config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                    "self_attention.num_heads": self.model.config.n_head // self.shard_config.tensor_parallel_size,
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attention.query_key_value",
                        target_module=ColCaiQuantLinear,
                        kwargs={"split_num": 3},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attention.dense", target_module=RowCaiQuantLinear, kwargs={"split_num": 1}
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attention.attention_dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.dense_h_to_4h", target_module=ColCaiQuantLinear, kwargs={"split_num": 1}
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.dense_4h_to_h", target_module=RowCaiQuantLinear, kwargs={"split_num": 1}
                    ),
                ],
            )
        # NOTE set inference mode to shard config
        self.shard_config._infer()

        # set as default, in inference we also use pipeline style forward, just setting stage as 1
        self.set_pipeline_forward(
            model_cls=BloomForCausalLM,
            new_forward=partial(
                BloomInferenceForwards.bloom_for_causal_lm_forward,
                tp_group=self.shard_config.tensor_parallel_process_group,
            ),
            policy=policy,
        )

        method_replacement = {"forward": BloomInferenceForwards.bloom_model_forward}
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=BloomModel)

        method_replacement = {"forward": BloomInferenceForwards.bloom_block_forward}
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=BloomBlock)

        method_replacement = {"forward": BloomInferenceForwards.bloom_attention_forward}
        self.append_or_create_method_replacement(
            description=method_replacement, policy=policy, target_key=BloomAttention
        )

        if HAS_TRITON_NORM:
            infer_method = get_triton_layernorm_forward()
            method_replacement = {"forward": partial(infer_method)}
            self.append_or_create_method_replacement(
                description=method_replacement, policy=policy, target_key=LayerNorm
            )

        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None

        if self.model.__class__.__name__ == "BloomModel":
            module = self.model
        else:
            module = self.model.transformer
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = self.distribute_layers(len(module.h), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.word_embeddings)
            held_layers.append(module.word_embeddings_layernorm)
            held_layers.append(self.model.lm_head)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.h[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.ln_f)

        return held_layers

from functools import partial
from typing import List

import torch
import torch.nn as nn

from colossalai.shardformer.modeling.chatglm2_6b.modeling_chatglm import (
    ChatGLMForConditionalGeneration,
    ChatGLMModel,
    GLMBlock,
    GLMTransformer,
    RMSNorm,
    SelfAttention,
)

# import colossalai
from colossalai.shardformer.policies.chatglm2 import ChatGLMForConditionalGenerationPolicy

from ..modeling._utils import init_to_get_rotary
from ..modeling.chatglm2 import ChatGLM2InferenceForwards

try:
    from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward as lightllm_rmsnorm_forward

    HAS_TRITON_RMSNORM = True
except:
    print("Did not find rms-norm triton kernel")
    print(
        "You can use the following command to install: pip install git+https://github.com/ModelTC/lightllm.git@ece7b43f8a6dfa74027adc77c2c176cff28c76c8"
    )
    HAS_TRITON_RMSNORM = False


def get_triton_rmsnorm_forward():
    if HAS_TRITON_RMSNORM:

        def _triton_rmsnorm_forward(self: RMSNorm, hidden_states: torch.Tensor):
            return lightllm_rmsnorm_forward(hidden_states, self.weight.data, self.eps)

        return _triton_rmsnorm_forward
    else:
        raise RuntimeError("Did not find rms-norm triton kernel")


class ChatGLM2InferPolicy(ChatGLMForConditionalGenerationPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()
        self.shard_config._infer()

        model_infer_forward = ChatGLM2InferenceForwards.chatglm_model_forward
        method_replacement = {"forward": model_infer_forward}
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=ChatGLMModel)

        encoder_infer_forward = ChatGLM2InferenceForwards.chatglm_encoder_forward
        method_replacement = {"forward": encoder_infer_forward}
        self.append_or_create_method_replacement(
            description=method_replacement, policy=policy, target_key=GLMTransformer
        )

        encoder_layer_infer_forward = ChatGLM2InferenceForwards.chatglm_glmblock_forward
        method_replacement = {"forward": encoder_layer_infer_forward}
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=GLMBlock)

        attn_infer_forward = ChatGLM2InferenceForwards.chatglm_flash_attn_kvcache_forward
        method_replacement = {"forward": attn_infer_forward}
        self.append_or_create_method_replacement(
            description=method_replacement, policy=policy, target_key=SelfAttention
        )
        if self.shard_config.enable_tensor_parallelism:
            policy[GLMBlock].attribute_replacement["self_attention.num_multi_query_groups_per_partition"] = (
                self.model.config.multi_query_group_num // self.shard_config.tensor_parallel_size
            )
        # for rmsnorm and others, we need to check the shape

        infer_forward = None

        if HAS_TRITON_RMSNORM:
            infer_forward = get_triton_rmsnorm_forward()
            if infer_forward is not None:
                method_replacement = {"forward": partial(infer_forward)}
                self.append_or_create_method_replacement(
                    description=method_replacement, policy=policy, target_key=RMSNorm
                )

        self.set_pipeline_forward(
            model_cls=ChatGLMForConditionalGeneration,
            new_forward=ChatGLM2InferenceForwards.chatglm_for_conditional_generation_forward,
            policy=policy,
        )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        module = self.model.transformer
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = self.distribute_layers(module.num_layers, stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.embedding)
            held_layers.append(module.output_layer)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.encoder.layers[start_idx:end_idx])
        if stage_manager.is_last_stage():
            if module.encoder.post_layer_norm:
                held_layers.append(module.encoder.final_layernorm)

        # rotary_pos_emb is needed for all stages
        held_layers.append(module.rotary_pos_emb)

        return held_layers

    def postprocess(self):
        init_to_get_rotary(self.model.transformer)
        return self.model

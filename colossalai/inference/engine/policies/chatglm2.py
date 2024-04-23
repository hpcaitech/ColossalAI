from typing import List

import torch.nn as nn

from colossalai.shardformer.modeling.chatglm2_6b.modeling_chatglm import (
    ChatGLMForConditionalGeneration,
    ChatGLMModel,
    GLMBlock,
    GLMTransformer,
    SelfAttention,
)

# import colossalai
from colossalai.shardformer.policies.chatglm2 import ChatGLMModelPolicy

from ..modeling._utils import init_to_get_rotary
from ..modeling.chatglm2 import ChatGLM2InferenceForwards

try:
    HAS_TRITON_RMSNORM = True
except:
    print("you should install triton from https://github.com/openai/triton")
    HAS_TRITON_RMSNORM = False


class ChatGLM2InferPolicy(ChatGLMModelPolicy):
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
        layers_per_stage = stage_manager.distribute_layers(module.num_layers)
        if stage_manager.is_first_stage():
            held_layers.append(module.embedding)
            held_layers.append(module.output_layer)
        start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
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

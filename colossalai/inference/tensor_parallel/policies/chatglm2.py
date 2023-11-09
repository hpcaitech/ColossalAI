from functools import partial

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
        return policy

    def postprocess(self):
        init_to_get_rotary(self.model)
        return self.model


class ChatGLM2ForConditionalGenerationInferPolicy(ChatGLM2InferPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()
        model_infer_forward = ChatGLM2InferenceForwards.chatglm_for_conditional_generation_forward
        method_replacement = {"forward": partial(model_infer_forward)}
        self.append_or_create_method_replacement(
            description=method_replacement, policy=policy, target_key=ChatGLMForConditionalGeneration
        )
        return policy

    def postprocess(self):
        return super().postprocess()

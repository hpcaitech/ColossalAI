from .bloom import BloomModelInferPolicy
from .chatglm2 import ChatGLM2InferPolicy
from .llama import LlamaModelInferPolicy

model_policy_map = {
    "llama": LlamaModelInferPolicy,
    "bloom": BloomModelInferPolicy,
    "chatglm": ChatGLM2InferPolicy,
}

__all__ = ["LlamaModelInferPolicy", "BloomModelInferPolicy", "ChatGLM2InferPolicy", "model_polic_map"]

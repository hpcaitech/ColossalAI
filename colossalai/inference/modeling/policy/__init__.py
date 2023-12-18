from .llama import LlamaModelInferPolicy

model_policy_map = {
    "llama": LlamaModelInferPolicy,
}

__all__ = ["LlamaModelInferPolicy", "model_polic_map"]

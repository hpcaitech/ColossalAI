from .nopadding_llama import NoPaddingLlamaModelInferPolicy
from .tp_nopadding_llama import TPNoPaddingLlamaModelInferPolicy

# from colossalai.shardformer.policies.llama import LlamaForCausalLMPolicy

model_policy_map = {
    "nopadding_llama": NoPaddingLlamaModelInferPolicy,
    "tp_llama": TPNoPaddingLlamaModelInferPolicy,
}

__all__ = ["NoPaddingLlamaModelInferPolicy", "model_polic_map", "TPNoPaddingLlamaModelInferPolicy"]

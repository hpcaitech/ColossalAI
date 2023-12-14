from colossalai.shardformer.policies.llama import LlamaForCausalLMPolicy


class LlamaModelInferPolicy(LlamaForCausalLMPolicy):
    # The code here just for Test and will be modified later.
    def __init__(self) -> None:
        super().__init__()

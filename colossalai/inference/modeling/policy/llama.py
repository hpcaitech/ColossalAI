from colossalai.shardformer.policies.llama import LlamaForCausalLMPolicy


class LlamaModelInferPolicy(LlamaForCausalLMPolicy):
    # The code here just for test and will be modified later.
    def __init__(self) -> None:
        super().__init__()

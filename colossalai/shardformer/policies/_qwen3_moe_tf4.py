__all__ = ["Qwen3MoePolicy", "Qwen3MoeModelPolicy", "Qwen3MoeForCausalLMPolicy"]

_MESSAGE = "Qwen3-MoE sharding needs transformers>=5.17 (the v4 layout has no fused experts)."


class Qwen3MoePolicy:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_MESSAGE)


class Qwen3MoeModelPolicy(Qwen3MoePolicy):
    pass


class Qwen3MoeForCausalLMPolicy(Qwen3MoePolicy):
    pass

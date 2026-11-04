__all__ = ["Qwen3MoeEPExperts"]

_MESSAGE = "Qwen3-MoE sharding needs transformers>=5.17 (the v4 layout has no fused experts)."


class Qwen3MoeEPExperts:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_MESSAGE)

    @staticmethod
    def from_native_module(*args, **kwargs):
        raise NotImplementedError(_MESSAGE)

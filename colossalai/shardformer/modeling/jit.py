import torch


def get_dropout_add_func():
    from transformers.models.bloom.modeling_bloom import dropout_add

    def self_dropout_add(self, x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
        return dropout_add(x, residual, prob, training)

    return self_dropout_add


def get_jit_fused_dropout_add_func():
    from colossalai.kernel.jit import bias_dropout_add_fused_inference, bias_dropout_add_fused_train

    def self_dropout_add(self, x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
        bias = torch.zeros_like(x)
        if training:
            return bias_dropout_add_fused_train(x, bias, residual, prob)
        return bias_dropout_add_fused_inference(x, bias, residual, prob)

    return self_dropout_add


def get_jit_fused_gelu_forward_func():
    from colossalai.kernel.jit.bias_gelu import bias_gelu

    def bloom_gelu_forward(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return bias_gelu(bias, x)

    return bloom_gelu_forward

"""This code is from NVIDIA apex:
      https://github.com/NVIDIA/apex
   with some changes. """

import numbers

import torch
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn import init
from torch.nn.parameter import Parameter


class FusedLayerNormAffineFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight, bias, normalized_shape, eps):
        try:
            from colossalai._C import layer_norm
        except ImportError:
            from colossalai.kernel.op_builder.layernorm import LayerNormBuilder
            layer_norm = LayerNormBuilder().load()

        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        bias_ = bias.contiguous()
        output, mean, invvar = layer_norm.forward_affine(input_, ctx.normalized_shape, weight_, bias_, ctx.eps)
        ctx.save_for_backward(input_, weight_, bias_, mean, invvar)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        try:
            from colossalai._C import layer_norm
        except ImportError:
            from colossalai.kernel.op_builder.layernorm import LayerNormBuilder
            layer_norm = LayerNormBuilder().load()

        input_, weight_, bias_, mean, invvar = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_input, grad_weight, grad_bias \
            = layer_norm.backward_affine(
                grad_output.contiguous(), mean, invvar,
                input_, ctx.normalized_shape,
                weight_, bias_, ctx.eps)

        return grad_input, grad_weight, grad_bias, None, None


class MixedFusedLayerNorm(torch.nn.Module):

    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None):
        super(MixedFusedLayerNorm, self).__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.empty(*normalized_shape, device=device, dtype=dtype))
        self.bias = Parameter(torch.empty(*normalized_shape, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):

        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input):

        return FusedLayerNormAffineFunction.apply(input, self.weight, self.bias, self.normalized_shape, self.eps)

    def __repr__(self):
        return f'MixedFusedLayerNorm(normalized_shape={self.normalized_shape}, eps={self.eps})'

# modified from torch-int: https://github.com/Guangxuan-Xiao/torch-int/blob/main/torch_int/nn/linear.py

import torch
from torch_int._CUDA import linear_a8_w8_b8_o8, linear_a8_w8_bfp32_ofp32
from torch_int.functional.quantization import quantize_per_tensor_absmax

try:
    from colossalai.kernel.op_builder.smoothquant import SmoothquantBuilder

    smoothquant_cuda = SmoothquantBuilder().load()
    HAS_SMOOTHQUANT_CUDA = True
except ImportError:
    HAS_SMOOTHQUANT_CUDA = False
    raise ImportError("CUDA smoothquant linear is not installed")


class W8A8BFP32O32LinearSiLU(torch.nn.Module):
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randint(
                -127,
                127,
                (self.out_features, self.in_features),
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "bias",
            torch.zeros((1, self.out_features), dtype=torch.float, requires_grad=False),
        )
        self.register_buffer("a", torch.tensor(alpha))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = smoothquant_cuda.linear_silu_a8_w8_bfp32_ofp32(x, self.weight, self.bias, self.a.item(), 1.0)
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale):
        int8_module = W8A8BFP32O32LinearSiLU(module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        alpha = input_scale * weight_scale
        int8_module.weight = int8_weight
        if module.bias is not None:
            int8_module.bias.data.copy_(module.bias.to(torch.float))
        int8_module.a = alpha
        return int8_module


# modified from torch-int: https://github.com/Guangxuan-Xiao/torch-int/blob/main/torch_int/nn/linear.py
class W8A8B8O8Linear(torch.nn.Module):
    # For qkv_proj
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randint(
                -127,
                127,
                (self.out_features, self.in_features),
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "bias",
            torch.zeros((1, self.out_features), dtype=torch.int8, requires_grad=False),
        )
        self.register_buffer("a", torch.tensor(alpha))
        self.register_buffer("b", torch.tensor(beta))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = linear_a8_w8_b8_o8(x, self.weight, self.bias, self.a.item(), self.b.item())
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale, output_scale):
        int8_module = W8A8B8O8Linear(module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        alpha = input_scale * weight_scale / output_scale
        int8_module.weight = int8_weight
        int8_module.a = alpha

        if module.bias is not None:
            int8_bias, bias_scale = quantize_per_tensor_absmax(module.bias)
            int8_module.bias = int8_bias
            beta = bias_scale / output_scale
            int8_module.b = beta

        return int8_module


# modified from torch-int: https://github.com/Guangxuan-Xiao/torch-int/blob/main/torch_int/nn/linear.py
class W8A8BFP32OFP32Linear(torch.nn.Module):
    # For fc2 and out_proj
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randint(
                -127,
                127,
                (self.out_features, self.in_features),
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "bias",
            torch.zeros(self.out_features, dtype=torch.float32, requires_grad=False),
        )
        self.register_buffer("a", torch.tensor(alpha))

    def _apply(self, fn):
        # prevent the bias from being converted to half
        super()._apply(fn)
        self.bias = self.bias.to(torch.float32)
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        self.bias = self.bias.to(torch.float32)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = linear_a8_w8_bfp32_ofp32(x, self.weight, self.bias, self.a.item(), 1)
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale):
        int8_module = W8A8BFP32OFP32Linear(module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        alpha = input_scale * weight_scale
        int8_module.weight = int8_weight
        int8_module.a = alpha
        int8_module.input_scale = input_scale
        int8_module.weight_scale = weight_scale

        if module.bias is not None:
            int8_module.bias = module.bias.to(torch.float32)

        return int8_module

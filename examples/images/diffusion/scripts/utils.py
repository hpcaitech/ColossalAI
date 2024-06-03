import bitsandbytes as bnb
import torch
import torch.nn as nn


class Linear8bit(nn.Linear):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        has_fp16_weights=False,
        memory_efficient_backward=False,
        threshold=6.0,
        weight_data=None,
        bias_data=None,
    ):
        super(Linear8bit, self).__init__(input_features, output_features, bias)
        self.state = bnb.MatmulLtState()
        self.bias = bias_data
        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        self.register_parameter("SCB", nn.Parameter(torch.empty(0), requires_grad=False))
        self.weight = weight_data
        self.quant()

    def quant(self):
        weight = self.weight.data.contiguous().half().cuda()
        CB, _, SCB, _, _ = bnb.functional.double_quant(weight)
        delattr(self, "weight")
        setattr(self, "weight", nn.Parameter(CB, requires_grad=False))
        delattr(self, "SCB")
        setattr(self, "SCB", nn.Parameter(SCB, requires_grad=False))
        del weight

    def forward(self, x):
        self.state.is_training = self.training

        if self.bias is not None and self.bias.dtype != torch.float16:
            self.bias.data = self.bias.data.half()

        self.state.CB = self.weight.data
        self.state.SCB = self.SCB.data

        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)
        del self.state.CxB
        return out


def replace_module(model):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_module(module)

        if isinstance(module, nn.Linear) and "out_proj" not in name:
            model._modules[name] = Linear8bit(
                input_features=module.in_features,
                output_features=module.out_features,
                threshold=6.0,
                weight_data=module.weight,
                bias_data=module.bias,
            )
    return model


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print("Model Size: {:.3f}MB".format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

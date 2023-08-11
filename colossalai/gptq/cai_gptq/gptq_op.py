
from ..config import CaiInferenceConfig
from ..inference_builder import inference_cuda_module
from .gptq_autotune import AutoTune
from .gptq_triton import gptq_fused_linear_triton
import torch
class BaseOp(torch.nn.Module):
    inference_cuda_module = inference_cuda_module
    def __init__(self, config: CaiInferenceConfig):
        super(BaseOp, self).__init__()
        self.config = config
        if BaseOp.inference_cuda_module is None:
            BaseOp.inference_cuda_module = inference_cuda_module


class CaiGPTQLinearOp(BaseOp):
    autotune = None

    def __init__(self, config: CaiInferenceConfig):
        super(CaiGPTQLinearOp, self).__init__(config)

        self.linear_func = self.inference_cuda_module.gptq_act_linear_fp16

        self.group_size = config.gptq_group_size
        self.bits = config.gptq_quant_bits
        self.maxq = 2**self.bits - 1
        self.empty_tensor = torch.zeros(4, device=torch.cuda.current_device())
        if CaiGPTQLinearOp.autotune == None:
            CaiGPTQLinearOp.autotune = AutoTune(self.linear_func)

    def forward(self,
                input: torch.Tensor,
                weight: torch.Tensor,
                weight_scales: torch.Tensor,
                weight_zeros: torch.Tensor,
                act_type = 0,
                bias: torch.Tensor = None,
                residual: torch.Tensor=None,
                qkv_fused = False):
        add_bias = True
        if bias is None:
            bias = self.empty_tensor
            add_bias = False

        add_residual = True
        if residual is None:
            residual = self.empty_tensor
            add_residual = False
        x = input.view(-1, input.shape[-1])

        if x.shape[0] > 1:
            out = gptq_fused_linear_triton(x, weight, weight_scales, weight_zeros, bias, residual,
                        self.bits, self.maxq, self.group_size, qkv_fused, add_bias, add_residual, act_type=act_type)
            if qkv_fused:
                out = out.view(3, input.shape[0], input.shape[1], weight.shape[-1])
            else:
                out = out.view(input.shape[0], input.shape[1], weight.shape[-1])
        else:
            print("inut shape, ", input.shape)
            config = {
                "input_dim": input.shape[-1],
                "input_len": input.shape[0] * input.shape[1] ,
                "add_bias": add_bias,
                "add_residual": add_residual,
                "qkv_fused": qkv_fused,
                "act_type": act_type,
                "out_dim": weight.shape[-1],
                "in_dim": input.shape[-1], 
                "wdtype": weight.dtype
            }

            best_config = CaiGPTQLinearOp.autotune.get_best_config(config,
                                    input,
                                    weight,
                                    weight_scales,
                                    weight_zeros,
                                    bias,
                                    residual,
                                    self.group_size,
                                    act_type,
                                    add_bias,
                                    add_residual,
                                    qkv_fused,
                                    128,
                                    128)
            block_size_x = best_config['linear_x']
            block_size_y = best_config['linear_y']
            out = self.linear_func(input,
                                    weight,
                                    weight_scales,
                                    weight_zeros,
                                    bias,
                                    residual,
                                    self.group_size,
                                    act_type,
                                    add_bias,
                                    add_residual,
                                    qkv_fused,
                                    block_size_x,
                                    block_size_y)
        return out
import torch

from colossalai.kernel.triton import gptq_fused_linear_triton


class CaiGPTQLinearOp(torch.nn.Module):
    def __init__(self, gptq_group_size, gptq_quant_bits):
        super(CaiGPTQLinearOp, self).__init__()
        self.group_size = gptq_group_size
        self.bits = gptq_quant_bits
        self.maxq = 2**self.bits - 1
        self.empty_tensor = torch.zeros(4, device=torch.cuda.current_device())

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scales: torch.Tensor,
        weight_zeros: torch.Tensor,
        g_idx: torch.Tensor = None,
        act_type=0,
        bias: torch.Tensor = None,
        residual: torch.Tensor = None,
        qkv_fused=False,
    ):
        add_bias = True
        if bias is None:
            bias = self.empty_tensor
            add_bias = False

        add_residual = True
        if residual is None:
            residual = self.empty_tensor
            add_residual = False
        x = input.view(-1, input.shape[-1])

        out = gptq_fused_linear_triton(
            x,
            weight,
            weight_scales,
            weight_zeros,
            bias,
            residual,
            self.bits,
            self.maxq,
            self.group_size,
            qkv_fused,
            add_bias,
            add_residual,
            act_type=act_type,
            g_idx=g_idx,
        )
        if qkv_fused:
            out = out.view(3, input.shape[0], input.shape[1], weight.shape[-1])
        else:
            out = out.view(input.shape[0], input.shape[1], weight.shape[-1])

        return out

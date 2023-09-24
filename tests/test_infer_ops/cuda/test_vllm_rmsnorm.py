#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import pytest
import torch
from torch import nn

try:
    from vllm import layernorm_ops

    rms_norm = layernorm_ops.rms_norm
    HAS_VLLM_KERNERL = True
except:
    print("please install vllm kernels to install rmsnorm")
    print("install vllm from https://github.com/vllm-project/vllm to accelerate your inference")
    HAS_VLLM_KERNERL = False


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def cuda_rmsnorm_forward(hidden_states, weight, variance_epsilon):
    x = hidden_states
    out = torch.empty_like(x)
    rms_norm(
        out,
        x,
        weight,
        variance_epsilon,
    )
    return out


@pytest.mark.skipif(not HAS_VLLM_KERNERL, reason="You need to install llama supported cuda kernels to run this test")
def test_rmsnorm():
    data = torch.randn((1024, 64), dtype=torch.float16, device="cuda")
    hg_rms = LlamaRMSNorm(64)
    hg_rms = hg_rms.half().cuda()
    out_torch = hg_rms(data)
    out_cuda = cuda_rmsnorm_forward(data, hg_rms.weight.data, hg_rms.variance_epsilon)

    check = torch.allclose(out_torch.cpu(), out_cuda.cpu(), rtol=1e-3, atol=1e-5)
    assert check is True, "cuda rmsnorm forward is not matched with torch rmsnorm forward"


if __name__ == "__main__":
    test_rmsnorm()

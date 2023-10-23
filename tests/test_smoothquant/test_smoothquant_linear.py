import warnings

import pytest
import torch

try:
    from colossalai.kernel.op_builder.smoothquant import SmoothquantBuilder

    smoothquant_cuda = SmoothquantBuilder().load()
    HAS_SMOOTHQUANT_CUDA = True
except:
    warnings.warn("CUDA smoothquant linear is not installed")
    HAS_SMOOTHQUANT_CUDA = False


@pytest.mark.skipif(
    not HAS_SMOOTHQUANT_CUDA,
    reason="smoothquant linear not installed properly",
)
def test_linear():
    a = torch.randint(-127, 127, (128, 512), dtype=torch.int8, device="cuda")
    b = torch.randint(-127, 127, (512, 256), dtype=torch.int8, device="cuda")
    c = torch.rand(256, dtype=torch.float, device="cuda")

    alpha = 1 / 127
    beta = 1.0
    torch_out = torch.mm(a.to(torch.float) * alpha, b.to(torch.float)) + c

    silu = torch.nn.SiLU()
    torch_out = silu(torch_out)

    b = b.transpose(0, 1).contiguous()
    cuda_out = smoothquant_cuda.linear_silu_a8_w8_bfp32_ofp32(a, b, c, alpha, beta)

    assert torch.allclose(torch_out, cuda_out, rtol=1e-02, atol=1e-02)


if __name__ == "__main__":
    test_linear()

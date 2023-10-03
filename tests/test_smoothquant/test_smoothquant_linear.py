import warnings

import pytest
import torch
from packaging import version

try:
    from colossalai.kernel.op_builder.smoothquant import SmoothquantBuilder

    smoothquant_cuda = SmoothquantBuilder().load()
    HAS_SMOOTHQUANT_CUDA = True
except ImportError:
    warnings.warn("CUDA gptq is not installed")
    HAS_SMOOTHQUANT_CUDA = False

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")


@pytest.mark.skipif(
    not TRITON_CUDA_SUPPORT or not HAS_SMOOTHQUANT_CUDA,
    reason="triton requires cuda version to be higher than 11.4",
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

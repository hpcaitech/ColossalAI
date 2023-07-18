import pytest
from packaging import version
import torch
from torch import nn

from colossalai.kernel.triton.ops import softmax

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse('11.4')

@pytest.mark.skipif(not TRITON_CUDA_SUPPORT, reason="triton requires cuda version to be higher than 11.4")
def test_softmax_op():
    data_samples = [
                        torch.randn((3, 4, 5, 32), device = "cuda", dtype = torch.float32),
                        torch.randn((320, 320, 78), device = "cuda", dtype = torch.float32),
                        torch.randn((2345, 4, 5, 64), device = "cuda", dtype = torch.float16)
                    ]

    for data in data_samples:
        module = nn.Softmax(dim = -1)
        data_torch_out = module(data)
        data_triton_out = softmax(data)
        check = torch.allclose(data_torch_out.cpu(), data_triton_out.cpu(), rtol=1e-3, atol=1e-3)
        assert check is True, "softmax outputs from triton and torch are not matched"


if __name__ == "__main__":
    test_softmax_op()
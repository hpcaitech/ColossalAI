import pytest
import torch
from torch import nn

from colossalai.kernel.triton.ops import softmax

def test_softmax_op():
    device = "cuda"
    data_samples = [
                        torch.randn((3, 4, 5, 32), device = "cuda", dtype = torch.float32),
                        torch.randn((320, 320, 78), device = "cuda", dtype = torch.float32),
                        torch.randn((2345, 4, 5, 64), device = "cuda", dtype = torch.float16)
                    ]

    for data in data_samples:
        module = nn.Softmax(dim = -1)
        data_torch_out = module(data)
        data_triton_out = softmax(data)
        print(torch.allclose(data_torch_out.cpu(), data_triton_out.cpu(), rtol=1e-3, atol=1e-3))


if __name__ == "__main__":
    test_softmax_op()
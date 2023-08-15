import os
import pytest
import numpy as np

import torch
from torch.nn import functional as F
try:
    from col_fused_softmax_lib import scaled_masked_softmax_forward
    HAS_INFER_CUDA = True
except:
    HAS_INFER_CUDA = False
    print("please install your cuda ")

@pytest.mark.skipif(not HAS_INFER_CUDA, reason="You need to install llama supported cuda kernels to run this test")    
def test():
    size = (17, 3, 1024, 256)
    data = torch.randn(size = size, device="cuda", dtype=torch.float16)
    mask = torch.zeros(size = (17, 1, 1024, 256), device="cuda", dtype=torch.uint8)

    out_cuda = scaled_masked_softmax_forward(data, mask, 1)

    out_torch = F.softmax(data, dim = -1)

    check = torch.allclose(out_cuda.cpu(), out_torch.cpu(), rtol=1e-3, atol=1e-3)
    assert check is True, "the output from cuda softmax is not matched with output from torch"

if __name__ == "__main__":
    test()
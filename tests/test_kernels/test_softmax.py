import pytest
import torch
from torch import nn

def test_softmax_op():
    device = "cuda"
    data = torch.randn((3, 4, 5, 32), device = "cuda", dtype = torch.float32)

    
if __name__ == "__main__":
    test_softmax_op()
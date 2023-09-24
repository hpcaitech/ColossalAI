# Adapted from ModelTC https://github.com/ModelTC/lightllm


import pytest
import torch
from packaging import version

try:
    pass

    from colossalai.kernel.triton.rotary_embedding_kernel import rotary_embedding_fwd

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")


def torch_rotary_emb(x, cos, sin):
    seq_len, h, dim = x.shape
    x0 = x[:, :, 0 : dim // 2]
    x1 = x[:, :, dim // 2 : dim]
    cos = cos.view((seq_len, 1, dim // 2))
    sin = sin.view((seq_len, 1, dim // 2))
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    return torch.cat((o0, o1), dim=-1)


@pytest.mark.skipif(
    not TRITON_CUDA_SUPPORT or not HAS_TRITON, reason="triton requires cuda version to be higher than 11.4"
)
def test_rotary_emb():
    SEQ_LEN = 1
    HEAD_NUM = 32
    HEAD_DIM = 128
    dtype = torch.half
    # create data
    x_shape = (SEQ_LEN, HEAD_NUM, HEAD_DIM)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")
    cos_shape = (SEQ_LEN, HEAD_DIM // 2)
    cos = -1.2 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    sin = -2.0 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    # forward pass
    y_torch = torch_rotary_emb(x, cos, sin)
    rotary_embedding_fwd(x, cos, sin)
    y_triton = x
    # compare
    assert torch.allclose(y_torch, y_triton, atol=1e-2, rtol=0)


if __name__ == "__main__":
    test_rotary_emb()

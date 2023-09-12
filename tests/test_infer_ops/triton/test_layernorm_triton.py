import pytest
import torch
from packaging import version

from colossalai.kernel.triton import layer_norm
from colossalai.testing.utils import parameterize

try:
    import triton
    import triton.language as tl

    from colossalai.kernel.triton.fused_layernorm import _layer_norm_fwd_fused
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse('11.4')


@pytest.mark.skipif(not TRITON_CUDA_SUPPORT or not HAS_TRITON,
                    reason="triton requires cuda version to be higher than 11.4")
@parameterize('M', [2, 4, 8, 16])
@parameterize('N', [64, 128])
def test_layer_norm(M, N):
    dtype = torch.float16
    eps = 1e-5
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device='cuda')
    bias = torch.rand(w_shape, dtype=dtype, device='cuda')
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')

    y_triton = layer_norm(x, weight, bias, eps)
    y_torch = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)

    assert y_triton.shape == y_torch.shape
    assert y_triton.dtype == y_torch.dtype
    print("max delta: ", torch.max(torch.abs(y_triton - y_torch)))
    assert torch.allclose(y_triton, y_torch, atol=1e-2, rtol=0)


if __name__ == "__main__":
    test_layer_norm()

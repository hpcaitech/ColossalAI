import pytest
import torch
from packaging import version
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from colossalai.kernel.triton import rms_layernorm
from colossalai.testing.utils import parameterize

try:
    import triton  # noqa

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")


@pytest.mark.skipif(
    not TRITON_CUDA_SUPPORT or not HAS_TRITON, reason="triton requires cuda version to be higher than 11.4"
)
@parameterize("M", [2, 4, 8, 16])
@parameterize("N", [64, 128])
def test_layer_norm(M, N):
    dtype = torch.float16
    eps = 1e-5
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.ones(w_shape, dtype=dtype, device="cuda")
    residual = torch.rand(x_shape, dtype=dtype, device="cuda")
    residual_copy = residual.clone()
    rms_norm = LlamaRMSNorm(hidden_size=N, eps=eps).cuda()
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")
    x_copy = x.clone()

    y_triton, _ = rms_layernorm(x, weight, eps=eps)
    y_llama = rms_norm.forward(x).to(dtype)

    assert y_triton.shape == y_llama.shape
    assert torch.allclose(y_triton, y_llama, atol=1e-5, rtol=1e-3)

    y_triton, residual = rms_layernorm(x, weight, eps=eps, residual=residual)

    x = x_copy + residual_copy

    y_llama = rms_norm.forward(x).to(dtype)

    assert y_triton.shape == y_llama.shape
    assert torch.allclose(y_triton, y_llama, atol=1e-5, rtol=1e-3)
    assert torch.allclose(x, residual, atol=1e-5, rtol=1e-3)


if __name__ == "__main__":
    test_layer_norm()

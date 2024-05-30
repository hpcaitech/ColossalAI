import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from colossalai.kernel.kernel_loader import InferenceOpsLoader
from colossalai.utils import get_current_device

inference_ops = InferenceOpsLoader().load()


@pytest.mark.parametrize("M", [2, 4, 8, 16])
@pytest.mark.parametrize("N", [64, 128, 512, 5120])
def test_rms_layernorm(M: int, N: int):
    torch.manual_seed(123)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    device = get_current_device()

    dtype = torch.float16
    eps = 1e-5
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.ones(w_shape, dtype=dtype, device=device)
    residual = torch.rand(x_shape, dtype=dtype, device=device)
    residual_copy = residual.clone()
    rms_norm = LlamaRMSNorm(hidden_size=N, eps=eps).cuda()
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")
    x_copy = x.clone()

    y_cuda = torch.empty_like(x)
    inference_ops.rms_layernorm(y_cuda, x, weight, eps)
    y_llama = rms_norm.forward(x).to(dtype)

    assert y_cuda.shape == y_llama.shape
    assert torch.allclose(y_cuda, y_llama, atol=1e-5, rtol=1e-3)

    inference_ops.fused_add_rms_layernorm(x, residual, weight, eps)
    y_cuda = x

    x = x_copy + residual_copy
    y_llama = rms_norm.forward(x).to(dtype)

    assert y_cuda.shape == y_llama.shape
    assert torch.allclose(y_cuda, y_llama, atol=1e-5, rtol=1e-3)
    assert torch.allclose(x, residual, atol=1e-5, rtol=1e-3)


if __name__ == "__main__":
    test_rms_layernorm(16, 5120)

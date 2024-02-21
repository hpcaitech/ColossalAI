import pytest
import torch
import triton
from packaging import version
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from vllm.model_executor.layers.layernorm import RMSNorm

from colossalai.kernel.triton import rms_layernorm
from colossalai.testing.utils import parameterize

try:
    pass

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


# Triton benchmark plot attributions
configs = [
    triton.testing.Benchmark(
        x_names=["SEQUENCE_TOTAL"],
        x_vals=[i for i in range(128, 1025, 128)],
        line_arg="provider",
        line_vals=[
            "vllm_rms_layernorm",
            "triton_rms_layernorm",
            "triton_rms_layernorm_with_residual",
            "vllm_rms_layernorm_with_residual",
        ],
        line_names=[
            "vllm_rms_layernorm",
            "triton_rms_layernorm",
            "triton_rms_layernorm_with_residual",
            "vllm_rms_layernorm_with_residual",
        ],
        styles=[("red", "-"), ("blue", "-"), ("yellow", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=f"RMSNorm benchmarking results",
        args={"HIDDEN_SIZE": 1024},
    )
]


@triton.testing.perf_report(configs)
def benchmark_rms_layernorm(
    provider: str,
    SEQUENCE_TOTAL: int,
    HIDDEN_SIZE: int,
):
    warmup = 10
    rep = 1000

    dtype = torch.float16
    eps = 1e-5
    x_shape = (SEQUENCE_TOTAL, HIDDEN_SIZE)
    w_shape = (x_shape[-1],)
    residual = torch.rand(x_shape, dtype=dtype, device="cuda")
    weight = torch.ones(w_shape, dtype=dtype, device="cuda")
    vllm_norm = RMSNorm(hidden_size=HIDDEN_SIZE, eps=eps).to(dtype=dtype, device="cuda")
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")
    if provider == "vllm_rms_layernorm":
        fn = lambda: vllm_norm(x)
    elif provider == "triton_rms_layernorm":
        fn = lambda: rms_layernorm(x, weight, eps=eps)
    elif provider == "vllm_rms_layernorm_with_residual":
        fn = lambda: vllm_norm(x, residual=residual)
    elif provider == "triton_rms_layernorm_with_residual":
        fn = lambda: rms_layernorm(x, weight, eps=eps, residual=residual)
    else:
        raise ValueError("Undefined provider.")

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    return ms


if __name__ == "__main__":
    test_layer_norm()
    # benchmark_rms_layernorm.run(save_path=".", print_data=True)

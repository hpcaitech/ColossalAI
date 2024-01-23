import pytest
import torch
from packaging import version
import triton

from colossalai.kernel.triton import rms_layernorm
from colossalai.testing.utils import parameterize
from transformers.models.llama.modeling_llama import LlamaRMSNorm

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
    rms_norm = LlamaRMSNorm(hidden_size=N, eps=eps).cuda()
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")

    y_triton = rms_layernorm(x, weight, eps=eps)
    y_llama = rms_norm.forward(x).to(dtype)

    assert torch.allclose(y_triton, y_llama, atol=1e-5, rtol=1e-5)



# Triton benchmark plot attributions
configs = [
    triton.testing.Benchmark(
        x_names=["SEQUENCE_TOTAL"],
        x_vals=[i for i in range(128, 1025, 128)],
        line_arg="provider",
        line_vals=["llama_rms_layernorm", "triton_rms_layernorm"],
        line_names=["llama_rms_layernorm", "triton_rms_layernorm"],
        styles=[("red", "-"), ("blue", "-")],
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
    rep = 100

    dtype = torch.float16
    eps = 1e-5
    x_shape = (SEQUENCE_TOTAL, HIDDEN_SIZE)
    w_shape = (x_shape[-1],)
    weight = torch.ones(w_shape, dtype=dtype, device="cuda")
    rms_norm = LlamaRMSNorm(hidden_size=HIDDEN_SIZE, eps=eps).cuda()
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")

    if provider == "llama_rms_layernorm":
        fn = lambda: rms_norm.forward(x).to(dtype)
    elif provider == "triton_rms_layernorm":
        fn = lambda: rms_layernorm(x, weight, eps=eps)
    else:
        raise ValueError("Undefined provider.")

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    return ms
    


if __name__ == "__main__":
    test_layer_norm()
    # benchmark_rms_layernorm.run(save_path=".")
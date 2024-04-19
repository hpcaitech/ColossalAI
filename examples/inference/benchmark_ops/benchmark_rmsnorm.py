import torch

from colossalai.kernel.kernel_loader import InferenceOpsLoader
from colossalai.kernel.triton import rms_layernorm

try:
    import triton  # noqa
except ImportError:
    print("please install triton from https://github.com/openai/triton")

inference_ops = InferenceOpsLoader().load()

# Triton benchmark plot attributions
configs = [
    triton.testing.Benchmark(
        x_names=["SEQUENCE_TOTAL"],
        x_vals=[i for i in range(128, 1025, 128)],
        line_arg="provider",
        line_vals=[
            "vllm_rms_layernorm",
            "triton_rms_layernorm",
            "cuda_rms_layernorm",
            "vllm_rms_layernorm_with_residual",
            "triton_rms_layernorm_with_residual",
            "cuda_rms_layernorm_with_residual",
        ],
        line_names=[
            "vllm_rms_layernorm",
            "triton_rms_layernorm",
            "cuda_rms_layernorm",
            "vllm_rms_layernorm_with_residual",
            "triton_rms_layernorm_with_residual",
            "cuda_rms_layernorm_with_residual",
        ],
        styles=[("red", "-"), ("blue", "-"), ("yellow", "-"), ("red", "--"), ("blue", "--"), ("yellow", "--")],
        ylabel="ms",
        plot_name=f"RMSNorm benchmarking results",
        args={"HIDDEN_SIZE": 5120},
    )
]


@triton.testing.perf_report(configs)
def benchmark_rms_layernorm(
    provider: str,
    SEQUENCE_TOTAL: int,
    HIDDEN_SIZE: int,
):
    try:
        from vllm.model_executor.layers.layernorm import RMSNorm
    except ImportError:
        raise ImportError("Please install vllm from https://github.com/vllm-project/vllm")

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
    elif provider == "cuda_rms_layernorm":
        out = torch.empty_like(x)
        fn = lambda: inference_ops.rms_layernorm(out, x, weight, eps)
    elif provider == "vllm_rms_layernorm_with_residual":
        fn = lambda: vllm_norm(x, residual=residual)
    elif provider == "triton_rms_layernorm_with_residual":
        fn = lambda: rms_layernorm(x, weight, eps=eps, residual=residual)
    elif provider == "cuda_rms_layernorm_with_residual":
        fn = lambda: inference_ops.fused_add_rms_layernorm(x, residual, weight, eps)
    else:
        raise ValueError("Undefined provider.")

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    return ms


if __name__ == "__main__":
    benchmark_rms_layernorm.run(save_path=".", print_data=True)

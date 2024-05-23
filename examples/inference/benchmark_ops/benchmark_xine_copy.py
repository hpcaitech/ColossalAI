import torch

from colossalai.kernel.triton import get_xine_cache
from tests.test_infer.test_kernels.triton.test_xine_copy import get_cos_sin

try:
    import triton  # noqa

except ImportError:
    print("please install triton from https://github.com/openai/triton")


configs = [
    triton.testing.Benchmark(
        x_names=["max_num_tokens"],
        x_vals=[2**i for i in range(6, 12)],
        line_arg="provider",
        line_vals=["torch_get_cos_sin", "triton_get_cos_sin"],
        line_names=["torch_get_cos_sin", "triton_get_cos_sin"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel="ms",
        plot_name="Get_cos-sin_func",
        args={"batch_size": 16, "head_dim": 256},
    )
]


@triton.testing.perf_report(configs)
def benchmark_get_xine_cache(
    provider: str,
    max_num_tokens: int,
    batch_size: int,
    head_dim: int,
):
    warmup = 10
    rep = 1000
    dtype = torch.float16
    cos_cache = torch.randn((8912, head_dim), dtype=dtype, device="cuda")
    sin_cache = torch.randn((8912, head_dim), dtype=dtype, device="cuda")
    lengths = torch.randint(2, max_num_tokens, (batch_size,), device="cuda")

    if provider == "torch_get_cos_sin":
        fn = lambda: get_cos_sin(lengths, cos_cache, sin_cache, is_prompts=True, dtype=dtype)
    elif provider == "triton_get_cos_sin":
        fn = lambda: get_xine_cache(lengths, cos_cache, sin_cache, is_prompts=True)
    else:
        raise ValueError("Undefined provider")

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms


if __name__ == "__main__":
    benchmark_get_xine_cache.run(save_path=".", print_data=True)

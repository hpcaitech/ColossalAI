import pytest
import torch
from packaging import version

from colossalai.inference.modeling.models.llama import get_cos_sin
from colossalai.kernel.triton import get_xine_cache

try:
    import triton  # noqa

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")


@pytest.mark.parametrize("BATCH_SIZE", [4])
@pytest.mark.parametrize("MAX_SEQ_LEN", [64])
@pytest.mark.parametrize("HEAD_DIM", [64])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_get_xine_cache(BATCH_SIZE, MAX_SEQ_LEN, HEAD_DIM, dtype):
    MAX_TOTAL_TOKENS = BATCH_SIZE * MAX_SEQ_LEN
    cos_cache = torch.randn((MAX_TOTAL_TOKENS, HEAD_DIM), dtype=dtype, device="cuda")
    lengths = torch.randint(2, MAX_SEQ_LEN, (BATCH_SIZE,), device="cuda")
    # prefill
    cos_ref, sin_ref = get_cos_sin(lengths, cos_cache, cos_cache, is_prompts=True, dtype=dtype)
    cos = get_xine_cache(lengths, cos_cache, is_prompts=True)
    assert torch.allclose(cos, cos_ref)
    # decoding
    cos_ref, sin_ref = get_cos_sin(lengths, cos_cache, cos_cache, is_prompts=False, dtype=dtype)
    lengths -= 1
    cos = get_xine_cache(lengths, cos_cache, is_prompts=False)

    assert torch.allclose(cos, cos_ref)


configs = [
    triton.testing.Benchmark(
        x_names=["max_num_tokens"],
        x_vals=[2**i for i in range(8, 12)],
        line_arg="provider",
        line_vals=["torch_get_cos_sin_func", "triton_get_xine_func"],
        line_names=["torch_get_cos_sin_func", "triton_get_xine_func"],
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
    rep = 100
    max_token_per_seq = max_num_tokens // batch_size
    dtype = torch.float16
    cos_cache = torch.randn((max_num_tokens, head_dim), dtype=dtype, device="cuda")
    sin_cache = torch.randn((max_num_tokens, head_dim), dtype=dtype, device="cuda")
    lengths = torch.randint(2, max_token_per_seq, (batch_size,), device="cuda")

    if provider == "torch_get_cos_sin_func":
        fn = lambda: get_cos_sin(lengths, cos_cache, sin_cache, is_prompts=False, dtype=dtype)
    elif provider == "triton_get_xine_func":
        fn = lambda: [
            get_xine_cache(lengths, cos_cache, is_prompts=False),
            get_xine_cache(lengths, sin_cache, is_prompts=False),
        ]
    else:
        raise ValueError("Undefined provider")

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms


if __name__ == "__main__":
    test_get_xine_cache(4, 64, 256, torch.float32)
    # benchmark_get_xine_cache.run(save_path=".",print_data=True)

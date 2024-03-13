import torch
import triton
from vllm._C import ops

from colossalai.kernel.kernel_loader import InferenceOpsLoader
from colossalai.kernel.triton import rotary_embedding

inference_ops = InferenceOpsLoader().load()

BATCH = 16
configs = [
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=[2**i for i in range(4, 12)],
        line_arg="provider",
        line_vals=["triton_func", "colossal_cuda_func", "vllm_cuda_func"],
        line_names=["triton_func", "colossal_cuda_func", "vllm_cuda_func"],
        styles=[("red", "-"), ("blue", "-"), ("yellow", "-")],
        ylabel="ms",
        plot_name=f"rotary_emb-batch-{BATCH}",
        args={"num_kv_heads": 16},
    )
]


def torch_rotary_emb(x, cos, sin):
    seq_len, h, dim = x.shape
    x0 = x[:, :, 0 : dim // 2]
    x1 = x[:, :, dim // 2 : dim]
    cos = cos.view((seq_len, 1, dim // 2))
    sin = sin.view((seq_len, 1, dim // 2))
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    return torch.cat((o0, o1), dim=-1)


@triton.testing.perf_report(configs)
def benchmark_rotary_emb(
    provider: str,
    num_tokens: int,
    num_kv_heads: int,
):
    warmup = 10
    rep = 100

    head_dim = 128
    dtype = torch.float16
    q_shape = (num_tokens, num_kv_heads, head_dim)
    q = -2.3 + 0.5 * torch.randn(q_shape, dtype=dtype, device="cuda")
    k_shape = (num_tokens, num_kv_heads, head_dim)
    k = -2.3 + 0.5 * torch.randn(k_shape, dtype=dtype, device="cuda")
    cos_shape = (4096, head_dim // 2)
    cos = -1.2 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    sin = -2.0 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")

    cos_sin = torch.stack((cos, sin), dim=1).contiguous()

    positions = torch.arange(num_tokens).cuda()

    if provider == "triton_func":
        fn = lambda: rotary_embedding(q, k, cos, sin)
    elif provider == "colossal_cuda_func":
        fn = lambda: inference_ops.rotary_embedding(q, k, cos, sin)
    elif provider == "vllm_cuda_func":
        q = q.view(num_tokens, -1)
        k = k.view(num_tokens, -1)
        fn = lambda: ops.rotary_embedding(positions, q, k, head_dim, cos_sin, True)
    else:
        raise ValueError("Undefined provider")

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms


if __name__ == "__main__":
    benchmark_rotary_emb.run(save_path=".", print_data=True)

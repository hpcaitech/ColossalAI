import torch
import triton

from colossalai.kernel.triton.fused_rotary_embedding import fused_rotary_embedding

BATCH = 16
configs = [
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=[2**i for i in range(4, 12)],
        line_arg="provider",
        line_vals=["torch_rotary_emb_func", "triton_rotary_emb_func"],
        line_names=["torch_rotary_emb_func", "triton_rotary_emb_func"],
        styles=[("red", "-"), ("blue", "-")],
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
    lengths = torch.tensor([3, 4, 6, 7], device="cuda")

    if provider == "torch_rotary_emb_func":
        fn = lambda: torch_rotary_emb(q, cos[:num_tokens], sin[:num_tokens])
    elif provider == "triton_rotary_emb_func":
        fn = lambda: fused_rotary_embedding(q, k, cos, sin, lengths)
    else:
        raise ValueError("Undefined provider")

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms


if __name__ == "__main__":
    benchmark_rotary_emb.run(save_path=".", print_data=True)

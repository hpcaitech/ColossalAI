import torch

from colossalai.kernel.triton import rms_layernorm, rotary_embedding

DTYPE = torch.float16
DEVICE = torch.device("cuda")


def rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = x.float() * torch.rsqrt(variance + eps)
    return (normalized * weight.float()).to(x.dtype)


def rotary_embedding_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half_dim = x.shape[-1] // 2
    x_first, x_second = x[..., :half_dim], x[..., half_dim:]
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return torch.cat((x_first * cos - x_second * sin, x_first * sin + x_second * cos), dim=-1)


def check_rms_layernorm() -> None:
    eps = 1e-5
    x = torch.randn((8, 128), device=DEVICE, dtype=DTYPE)
    weight = torch.randn((128,), device=DEVICE, dtype=DTYPE)

    expected = rms_norm_reference(x, weight, eps)
    actual, _ = rms_layernorm(x, weight, eps=eps)

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def check_rotary_embedding() -> None:
    num_tokens, num_q_heads, num_kv_heads, head_dim = 8, 4, 2, 64
    q = torch.randn((num_tokens, num_q_heads, head_dim), device=DEVICE, dtype=DTYPE)
    k = torch.randn((num_tokens, num_kv_heads, head_dim), device=DEVICE, dtype=DTYPE)

    positions = torch.arange(num_tokens, device=DEVICE, dtype=torch.float32)[:, None]
    frequencies = torch.arange(0, head_dim, 2, device=DEVICE, dtype=torch.float32) / head_dim
    angles = positions / (10000**frequencies)[None, :]
    cos, sin = angles.cos().to(DTYPE), angles.sin().to(DTYPE)

    expected_q = rotary_embedding_reference(q, cos, sin)
    expected_k = rotary_embedding_reference(k, cos, sin)

    rotary_embedding(q, k, cos, sin)

    torch.testing.assert_close(q, expected_q, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(k, expected_k, rtol=1e-3, atol=1e-3)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_ops smoke test requires CUDA")

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    check_rms_layernorm()
    check_rotary_embedding()
    torch.cuda.synchronize()
    print("benchmark_ops smoke test passed: RMSNorm and Rotary Embedding match their PyTorch references")


if __name__ == "__main__":
    main()

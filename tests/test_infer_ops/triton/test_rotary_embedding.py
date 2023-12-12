import pytest
import torch

from colossalai.kernel.triton import rotary_embedding_fwd


def torch_rotary_emb(x, cos, sin):
    seq_len, h, dim = x.shape
    x0 = x[:, :, 0 : dim // 2]
    x1 = x[:, :, dim // 2 : dim]
    cos = cos.view((seq_len, 1, dim // 2))
    sin = sin.view((seq_len, 1, dim // 2))
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    return torch.cat((o0, o1), dim=-1)


@pytest.mark.parametrize("TOTAL_TOKENS", [64])
@pytest.mark.parametrize("H", [32])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_rotary_emb(TOTAL_TOKENS, H, D, dtype, eps=1e-5, device="cuda"):
    # create data
    q_shape = (TOTAL_TOKENS, H, D)
    q = -2.3 + 0.5 * torch.randn(q_shape, dtype=dtype, device="cuda")
    k_shape = (TOTAL_TOKENS, H, D)
    k = -2.3 + 0.5 * torch.randn(k_shape, dtype=dtype, device="cuda")
    cos_shape = (TOTAL_TOKENS, D // 2)
    cos = -1.2 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    sin = -2.0 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")

    q_ref = torch_rotary_emb(q, cos, sin)
    k_ref = torch_rotary_emb(k, cos, sin)

    rotary_embedding_fwd(q, k, cos, sin)
    q_tri = q
    k_tri = k

    assert torch.allclose(q_tri, q_ref, atol=1e-2, rtol=1e-3)
    assert torch.allclose(k_tri, k_ref, atol=1e-2, rtol=1e-3)


if __name__ == "__main__":
    test_rotary_emb(1024, 32, 64, torch.float16)

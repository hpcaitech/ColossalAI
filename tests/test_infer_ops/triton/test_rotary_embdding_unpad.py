import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

from colossalai.kernel.triton import rotary_embedding


def torch_rotary_emb(x, cos, sin):
    seq_len, h, dim = x.shape
    x0 = x[:, :, 0 : dim // 2]
    x1 = x[:, :, dim // 2 : dim]
    cos = cos.view((seq_len, 1, dim // 2))
    sin = sin.view((seq_len, 1, dim // 2))
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    return torch.cat((o0, o1), dim=-1)


@pytest.mark.parametrize("BATCH_SIZE", [4])
@pytest.mark.parametrize("SEQ_LEN", [64])
@pytest.mark.parametrize("H", [32])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_rotary_emb(BATCH_SIZE, SEQ_LEN, H, D, dtype):
    TOTAL_TOKENS = BATCH_SIZE * SEQ_LEN
    # our crafted op equals to Transformers
    x0 = torch.randn(TOTAL_TOKENS, SEQ_LEN, D)
    x1 = torch.randn(TOTAL_TOKENS, SEQ_LEN, D)
    emb = LlamaRotaryEmbedding(D)
    cos, sin = emb(x0, TOTAL_TOKENS)
    cos_2 = cos[:, :32]
    sin_2 = sin[:, :32]
    position_ids = torch.arange(TOTAL_TOKENS)
    embd_x0, _ = apply_rotary_pos_emb(x0, x1, cos, sin, position_ids)
    embd_stimulated_x = torch_rotary_emb(x0, cos_2, sin_2)
    assert torch.allclose(embd_x0, embd_stimulated_x)

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
    rotary_embedding(q, k, cos, sin)

    assert torch.allclose(q, q_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(k, k_ref, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_rotary_emb(4, 64, 32, 64, torch.float32)

import pytest
import torch
from packaging import version
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

from colossalai.kernel.triton import decoding_fused_rotary_embedding
from tests.test_infer.test_ops.triton.kernel_utils import mock_alloc_block_table_and_kvcache_v2

try:
    import triton  # noqa

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")


def torch_rotary_emb(x, cos, sin):
    seq_len, h, dim = x.shape
    x0 = x[:, :, 0 : dim // 2]
    x1 = x[:, :, dim // 2 : dim]
    cos = cos.view((seq_len, 1, dim // 2))
    sin = sin.view((seq_len, 1, dim // 2))
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    return torch.cat((o0, o1), dim=-1)


@pytest.mark.skipif(
    not TRITON_CUDA_SUPPORT or not HAS_TRITON, reason="triton requires cuda version to be higher than 11.4"
)
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
    block_size = 32
    max_num_blocks_per_seq = 4
    q_shape = (TOTAL_TOKENS, H, D)
    q = -2.3 + 0.5 * torch.randn(q_shape, dtype=dtype, device="cuda")
    k_shape = (TOTAL_TOKENS, H, D)
    k = -2.3 + 0.5 * torch.randn(k_shape, dtype=dtype, device="cuda")
    cos_shape = (TOTAL_TOKENS, D // 2)
    cos = -1.2 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    sin = -2.0 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    cache_shape = (BATCH_SIZE * max_num_blocks_per_seq, H, block_size, D)
    k_cache = torch.zeros(size=cache_shape, dtype=dtype, device="cuda")
    v = torch.randn_like(k)
    v_cache = torch.zeros_like(k_cache)
    past_kv_seq_lengths = torch.tensor([SEQ_LEN - 1 for _ in range(BATCH_SIZE)], dtype=torch.int32, device="cuda")
    block_tables = mock_alloc_block_table_and_kvcache_v2(
        k, v, k_cache, v_cache, past_kv_seq_lengths, BATCH_SIZE, max_num_blocks_per_seq, block_size
    )
    new_k = torch.randn((BATCH_SIZE, H, D), dtype=dtype, device="cuda")
    new_q = torch.randn_like(new_k)
    new_v = torch.randn_like(new_k)

    kv_seq_lengths = past_kv_seq_lengths + 1
    block_tables = block_tables.to(device="cuda")
    q_ref = torch_rotary_emb(new_q, cos[:BATCH_SIZE], sin[:BATCH_SIZE])

    decoding_fused_rotary_embedding(new_q, new_k, new_v, cos, sin, k_cache, v_cache, block_tables, kv_seq_lengths)
    assert torch.allclose(new_q, q_ref, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_rotary_emb(4, 64, 32, 64, torch.float32)

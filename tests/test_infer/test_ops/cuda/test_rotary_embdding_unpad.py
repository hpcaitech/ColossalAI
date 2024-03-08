import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

from colossalai.kernel.kernel_loader import InferenceOpsLoader

inference_ops = InferenceOpsLoader().load()

from tests.test_infer.test_ops.triton.kernel_utils import mock_alloc_block_table_and_kvcache_v2
from tests.test_infer.test_ops.triton.test_rotary_embdding_unpad import torch_rotary_emb


@pytest.mark.parametrize("BATCH_SIZE", [4])
@pytest.mark.parametrize("SEQ_LEN", [64])
@pytest.mark.parametrize("H", [32])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_rotary_emb(BATCH_SIZE, SEQ_LEN, H, D, dtype):
    torch.manual_seed(10)
    TOTAL_TOKENS = BATCH_SIZE * SEQ_LEN
    # our crafted op equals to Transformers
    x0 = torch.randn(TOTAL_TOKENS, SEQ_LEN, D, dtype=dtype)
    x1 = torch.randn(TOTAL_TOKENS, SEQ_LEN, D, dtype=dtype)
    emb = LlamaRotaryEmbedding(D)
    cos, sin = emb(x0, TOTAL_TOKENS)
    cos_2 = cos[:, : D // 2]
    sin_2 = sin[:, : D // 2]
    position_ids = torch.arange(TOTAL_TOKENS)
    embd_x0, _ = apply_rotary_pos_emb(x0, x1, cos, sin, position_ids)
    embd_stimulated_x = torch_rotary_emb(x0, cos_2, sin_2)
    assert torch.allclose(embd_x0, embd_stimulated_x)

    # create data
    block_size = 32
    max_blocks_per_sequence = (TOTAL_TOKENS + block_size - 1) // block_size
    q_shape = (TOTAL_TOKENS, H, D)
    q = -2.3 + 0.5 * torch.randn(q_shape, dtype=dtype, device="cuda")
    k_shape = (TOTAL_TOKENS, H, D)
    k = -2.3 + 0.5 * torch.randn(k_shape, dtype=dtype, device="cuda")
    cos_shape = (TOTAL_TOKENS, D // 2)
    cos = -1.2 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    sin = -2.0 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    cache_shape = (BATCH_SIZE * max_blocks_per_sequence, H, block_size, D)
    k_cache = torch.zeros(size=cache_shape, dtype=dtype, device="cuda")
    v = torch.randn_like(k)
    v_cache = torch.zeros_like(k_cache)
    past_kv_seq_lengths = torch.tensor([SEQ_LEN - 1 for _ in range(BATCH_SIZE)], dtype=torch.int32, device="cuda")
    block_tables = mock_alloc_block_table_and_kvcache_v2(
        k, v, k_cache, v_cache, past_kv_seq_lengths, BATCH_SIZE, max_blocks_per_sequence, block_size
    )
    new_k = torch.randn((BATCH_SIZE, H, D), dtype=dtype, device="cuda")
    new_q = torch.randn_like(new_k)
    new_v = torch.randn_like(new_k)

    kv_seq_lengths = past_kv_seq_lengths + 1
    block_tables = block_tables.to(device="cuda")
    q_ref = torch_rotary_emb(new_q, cos[:BATCH_SIZE], sin[:BATCH_SIZE])
    k_ref = torch_rotary_emb(new_k, cos[:BATCH_SIZE], sin[:BATCH_SIZE])

    new_q_copy = new_q.clone()
    new_k_copy = new_k.clone()

    inference_ops.rotary_embedding_and_cache_copy(
        new_q, new_k, new_v, cos, sin, k_cache, v_cache, kv_seq_lengths, block_tables
    )

    inference_ops.rotary_embedding(new_q_copy, new_k_copy, cos, sin)

    past_kv_seq_len = kv_seq_lengths - 1
    target_block_ids = block_tables[range(0, block_tables.size(0)), past_kv_seq_len // block_size]
    offsets_in_block = past_kv_seq_len % block_size
    k_target = k_cache[target_block_ids, :, offsets_in_block, :].squeeze()
    k_source = new_k_copy.squeeze()
    v_target = v_cache[target_block_ids, :, offsets_in_block, :].squeeze()
    v_source = new_v.squeeze()

    assert torch.allclose(new_q, q_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(k_target, k_ref, atol=1e-6, rtol=1e-6)

    assert torch.allclose(new_q_copy, q_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(new_k_copy, k_ref, atol=1e-6, rtol=1e-6)

    assert k_target.shape == k_source.shape
    assert torch.allclose(k_target, k_source, atol=1e-6, rtol=1e-6)

    assert v_target.shape == v_source.shape
    assert torch.equal(v_target, v_source)


if __name__ == "__main__":
    test_rotary_emb(16, 512, 4, 128, torch.float16)

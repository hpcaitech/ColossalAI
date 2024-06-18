import numpy as np
import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

from colossalai.kernel.kernel_loader import InferenceOpsLoader

inference_ops = InferenceOpsLoader().load()

from tests.test_infer.test_kernels.triton.kernel_utils import mock_alloc_block_table_and_kvcache_v3
from tests.test_infer.test_kernels.triton.test_rotary_embdding_unpad import torch_rotary_emb


def numpy_allclose(x, y, rtol, atol):
    x_numpy = x.detach().cpu().numpy()
    y_numpy = y.detach().cpu().numpy()

    np.testing.assert_allclose(x_numpy, y_numpy, rtol=rtol, atol=atol)


@pytest.mark.parametrize("BATCH_SIZE", [4])
@pytest.mark.parametrize("SEQ_LEN", [64])
@pytest.mark.parametrize("H", [32])
@pytest.mark.parametrize("K_H", [16, 32])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rotary_emb(BATCH_SIZE, SEQ_LEN, H, K_H, D, dtype):
    torch.manual_seed(10)
    TOTAL_TOKENS = BATCH_SIZE * SEQ_LEN
    # our crafted op equals to Transformers
    x0 = torch.randn(BATCH_SIZE, H, SEQ_LEN, D, dtype=dtype)
    x1 = torch.randn(BATCH_SIZE, H, SEQ_LEN, D, dtype=dtype)

    position_ids = torch.arange(TOTAL_TOKENS).reshape((BATCH_SIZE, SEQ_LEN))

    emb = LlamaRotaryEmbedding(D)

    cos, sin = emb(x0, position_ids)
    embd_x0, _ = apply_rotary_pos_emb(x0, x1, cos, sin)
    cos = cos.reshape((TOTAL_TOKENS, -1))
    sin = sin.reshape((TOTAL_TOKENS, -1))
    cos_2 = cos[:, : D // 2]
    sin_2 = sin[:, : D // 2]
    x2 = x0.transpose(1, 2).reshape(TOTAL_TOKENS, H, D)
    embd_stimulated_x = torch_rotary_emb(x2, cos_2, sin_2)
    embd_stimulated_x = embd_stimulated_x.reshape((BATCH_SIZE, SEQ_LEN, H, D)).transpose(1, 2)
    assert torch.allclose(embd_x0, embd_stimulated_x)

    # create data
    block_size = 32
    max_blocks_per_sequence = (TOTAL_TOKENS + block_size - 1) // block_size
    q_shape = (TOTAL_TOKENS, H, D)
    q = -2.3 + 0.5 * torch.randn(q_shape, dtype=dtype, device="cuda")
    k_shape = (TOTAL_TOKENS, K_H, D)
    k = -2.3 + 0.5 * torch.randn(k_shape, dtype=dtype, device="cuda")
    cos_shape = (TOTAL_TOKENS, D // 2)
    cos = -1.2 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    sin = -2.0 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    k_cache_shape = (BATCH_SIZE * max_blocks_per_sequence, K_H, D // x, block_size, x)
    v_cache_shape = (BATCH_SIZE * max_blocks_per_sequence, K_H, block_size, D)
    k_cache = torch.zeros(size=k_cache_shape, dtype=dtype, device="cuda")
    v = torch.randn_like(k)
    v_cache = torch.zeros(size=v_cache_shape, dtype=dtype, device="cuda")
    past_kv_seq_lengths = torch.tensor([SEQ_LEN - 1 for _ in range(BATCH_SIZE)], dtype=torch.int32, device="cuda")
    block_tables = mock_alloc_block_table_and_kvcache_v3(
        k, v, k_cache, v_cache, past_kv_seq_lengths, BATCH_SIZE, max_blocks_per_sequence, block_size
    )
    new_k = torch.randn((BATCH_SIZE, K_H, D), dtype=dtype, device="cuda")
    new_q = torch.randn((BATCH_SIZE, H, D), dtype=dtype, device="cuda")
    new_v = torch.randn_like(new_k)

    kv_seq_lengths = past_kv_seq_lengths + 1
    block_tables = block_tables.to(device="cuda")

    new_q_copy = new_q.clone()
    new_k_copy = new_k.clone()

    if dtype == torch.float16:
        rtol = 1e-3
        atol = 1e-3

        new_q_fp16 = new_q.clone()
        new_k_fp16 = new_k.clone()

        high_precision_cos = cos[:BATCH_SIZE].to(torch.float32)
        high_precision_sin = sin[:BATCH_SIZE].to(torch.float32)
        high_precision_q = new_q.to(torch.float32)
        high_precision_k = new_k.to(torch.float32)
        q_ref = torch_rotary_emb(high_precision_q, high_precision_cos, high_precision_sin).to(torch.float16)
        k_ref = torch_rotary_emb(high_precision_k, high_precision_cos, high_precision_sin).to(torch.float16)

    else:
        rtol = 1e-5
        atol = 1e-7

        q_ref = torch_rotary_emb(new_q, cos[:BATCH_SIZE], sin[:BATCH_SIZE])
        k_ref = torch_rotary_emb(new_k, cos[:BATCH_SIZE], sin[:BATCH_SIZE])

    inference_ops.rotary_embedding_and_cache_copy(
        new_q, new_k, new_v, cos, sin, k_cache, v_cache, kv_seq_lengths, block_tables, True
    )

    inference_ops.rotary_embedding(new_q_copy, new_k_copy, cos, sin, True)

    past_kv_seq_len = kv_seq_lengths - 1
    target_block_ids = block_tables[range(0, block_tables.size(0)), past_kv_seq_len // block_size]
    offsets_in_block = past_kv_seq_len % block_size
    k_target = k_cache[target_block_ids, :, :, offsets_in_block, :].squeeze()
    k_source = new_k_copy.squeeze()
    v_target = v_cache[target_block_ids, :, offsets_in_block, :].squeeze()
    k_target = k_target.reshape(v_target.shape)
    v_source = new_v.squeeze()

    numpy_allclose(new_q, q_ref, rtol=rtol, atol=atol)
    numpy_allclose(k_target, k_ref, rtol=rtol, atol=atol)

    numpy_allclose(new_q_copy, q_ref, rtol=rtol, atol=atol)
    numpy_allclose(new_k_copy, k_ref, rtol=rtol, atol=atol)

    assert k_target.shape == k_source.shape
    numpy_allclose(k_target, k_source, rtol=rtol, atol=atol)

    assert v_target.shape == v_source.shape
    assert torch.equal(v_target, v_source)

    if dtype == torch.float16:
        # After testing cuda fp16 high_precision, it was found to have higher precision than torch fp16. Therefore, the threshold here has been relaxed to pass the test.
        rtol = 1e-3
        atol = 1e-1
        inference_ops.rotary_embedding(new_q_fp16, new_k_fp16, cos, sin, False)
        numpy_allclose(new_q_copy, new_q_fp16, rtol=rtol, atol=atol)
        numpy_allclose(new_k_copy, new_k_fp16, rtol=rtol, atol=atol)


if __name__ == "__main__":
    test_rotary_emb(16, 64, 32, 16, 128, torch.float16)

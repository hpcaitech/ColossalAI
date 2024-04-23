from itertools import product

import numpy as np
import pytest
import torch

from colossalai.kernel.kernel_loader import InferenceOpsLoader
from colossalai.utils import get_current_device

inference_ops = InferenceOpsLoader().load()

from tests.test_infer.test_ops.triton.kernel_utils import (
    convert_kv_unpad_to_padded,
    create_attention_mask,
    generate_caches_and_block_tables_v2,
    generate_caches_and_block_tables_vllm,
    torch_attn_ref,
)

q_len = 1


def prepare_data(
    BATCH_SIZE: int,
    HEAD_SIZE: int,
    NUM_ATTN_HEADS: int,
    NUM_KV_HEADS: int,
    MAX_SEQ_LEN: int,
    dtype=torch.float16,
    device="cuda",
):
    # Use the provided maximum sequence length for each sequence when testing with teh same context length,
    # otherwise generate random context lengths.
    # returns
    #   q [BATCH_SIZE, NUM_ATTN_HEADS, HEAD_SIZE]
    #   k_unpad/v_unpad [num_tokens, NUM_KV_HEADS, HEAD_SIZE]
    kv_lengths = torch.randint(low=1, high=MAX_SEQ_LEN, size=(BATCH_SIZE,), dtype=torch.int32, device=device)
    num_tokens = torch.sum(kv_lengths).item()

    q_size = (BATCH_SIZE, q_len, NUM_ATTN_HEADS, HEAD_SIZE)
    q = torch.empty(size=q_size, dtype=dtype, device=device).normal_(mean=0.0, std=0.5).transpose(1, 2)
    kv_size = (num_tokens, 2 * NUM_KV_HEADS, HEAD_SIZE)
    kv_unpad = torch.empty(size=kv_size, dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    k_unpad, v_unpad = torch.split(kv_unpad, [NUM_KV_HEADS, NUM_KV_HEADS], dim=-2)

    return q, k_unpad, v_unpad, kv_lengths


def numpy_allclose(x, y, rtol, atol):
    x_numpy = x.detach().cpu().numpy()
    y_numpy = y.detach().cpu().numpy()

    np.testing.assert_allclose(x_numpy, y_numpy, rtol=rtol, atol=atol)


@pytest.mark.parametrize("BATCH_SIZE", [1, 4, 7, 32])
@pytest.mark.parametrize("BLOCK_SIZE", [8, 16, 32])
@pytest.mark.parametrize("MAX_NUM_BLOCKS_PER_SEQ", [1, 8, 32])
@pytest.mark.parametrize("HEAD_SIZE", [64, 128])
@pytest.mark.parametrize("NUM_ATTN_HEADS", [16])
@pytest.mark.parametrize("KV_GROUP_NUM", [1, 2, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_flash_decoding_attention(
    BATCH_SIZE, BLOCK_SIZE, MAX_NUM_BLOCKS_PER_SEQ, HEAD_SIZE, NUM_ATTN_HEADS, KV_GROUP_NUM, dtype
):
    torch.manual_seed(123)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    NUM_KV_HEADS = NUM_ATTN_HEADS // KV_GROUP_NUM
    assert isinstance(NUM_KV_HEADS, int) and NUM_KV_HEADS > 0, "Invalid number of kv heads."
    MAX_SEQ_LEN = BLOCK_SIZE * MAX_NUM_BLOCKS_PER_SEQ
    device = get_current_device()

    q, k_unpad, v_unpad, kv_seq_lengths = prepare_data(
        BATCH_SIZE, HEAD_SIZE, NUM_ATTN_HEADS, NUM_KV_HEADS, MAX_SEQ_LEN, dtype, device
    )

    k_cache, v_cache, block_tables = generate_caches_and_block_tables_v2(
        k_unpad, v_unpad, kv_seq_lengths, BATCH_SIZE, MAX_NUM_BLOCKS_PER_SEQ, BLOCK_SIZE, dtype, device
    )

    block_tables = block_tables.to(device=device)
    max_seq_len_across_batch = kv_seq_lengths.max().item()
    kv_max_split_num = (max_seq_len_across_batch + BLOCK_SIZE - 1) // BLOCK_SIZE
    output = torch.empty((BATCH_SIZE, NUM_ATTN_HEADS, HEAD_SIZE), dtype=dtype, device=device)
    sm_scale = 1.0 / (HEAD_SIZE**0.5)

    k_torch = convert_kv_unpad_to_padded(k_unpad, kv_seq_lengths, BATCH_SIZE, max_seq_len_across_batch)
    v_torch = convert_kv_unpad_to_padded(v_unpad, kv_seq_lengths, BATCH_SIZE, max_seq_len_across_batch)
    torch_padding_mask = create_attention_mask(kv_seq_lengths, BATCH_SIZE, q_len, max_seq_len_across_batch, device)

    mid_output = torch.empty(
        size=(BATCH_SIZE, NUM_ATTN_HEADS, kv_max_split_num, HEAD_SIZE), dtype=torch.float32, device=device
    )
    mid_output_lse = torch.empty(
        size=(BATCH_SIZE, NUM_ATTN_HEADS, kv_max_split_num), dtype=torch.float32, device=device
    )

    if dtype == torch.float16:
        rtol = 1e-3
        atol = 1e-3

        high_precision_q = q.to(torch.float32)
        high_precision_k_torch = k_torch.to(torch.float32)
        high_precision_v_torch = v_torch.to(torch.float32)
        out_ref = torch_attn_ref(
            high_precision_q,
            high_precision_k_torch,
            high_precision_v_torch,
            torch_padding_mask,
            BATCH_SIZE,
            q_len,
            max_seq_len_across_batch,
            NUM_ATTN_HEADS,
            NUM_KV_HEADS,
            HEAD_SIZE,
        ).to(torch.float16)

    else:
        rtol = 1e-5
        atol = 1e-7

        out_ref = torch_attn_ref(
            q,
            k_torch,
            v_torch,
            torch_padding_mask,
            BATCH_SIZE,
            q_len,
            max_seq_len_across_batch,
            NUM_ATTN_HEADS,
            NUM_KV_HEADS,
            HEAD_SIZE,
        )

    inference_ops.flash_decoding_attention(
        output,
        q.squeeze(2),
        k_cache,
        v_cache,
        kv_seq_lengths,
        block_tables,
        BLOCK_SIZE,
        max_seq_len_across_batch,
        mid_output,
        mid_output_lse,
        sm_scale,
    )
    numpy_allclose(out_ref, output, rtol=rtol, atol=atol)


try:
    from vllm._C import ops as vllm_ops  # noqa

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    print("The subsequent test requires vllm. Please refer to https://github.com/vllm-project/vllm")


@pytest.mark.skipif(not HAS_VLLM, reason="requires vllm")
@pytest.mark.parametrize("BATCH_SIZE", [1, 4, 7, 32])
@pytest.mark.parametrize("BLOCK_SIZE", [8, 16, 32])
@pytest.mark.parametrize("MAX_NUM_BLOCKS_PER_SEQ", [1, 8, 32])
@pytest.mark.parametrize("HEAD_SIZE", [64, 128])
@pytest.mark.parametrize("NUM_ATTN_HEADS", [16])
@pytest.mark.parametrize("KV_GROUP_NUM", [1, 2, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_vllm_flash_decoding_attention(
    BATCH_SIZE, BLOCK_SIZE, MAX_NUM_BLOCKS_PER_SEQ, HEAD_SIZE, NUM_ATTN_HEADS, KV_GROUP_NUM, dtype
):
    torch.manual_seed(123)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    NUM_KV_HEADS = NUM_ATTN_HEADS // KV_GROUP_NUM
    assert isinstance(NUM_KV_HEADS, int) and NUM_KV_HEADS > 0, "Invalid number of kv heads."
    MAX_SEQ_LEN = BLOCK_SIZE * MAX_NUM_BLOCKS_PER_SEQ
    device = get_current_device()

    q, k_unpad, v_unpad, kv_seq_lengths = prepare_data(
        BATCH_SIZE, HEAD_SIZE, NUM_ATTN_HEADS, NUM_KV_HEADS, MAX_SEQ_LEN, dtype, device
    )

    k_cache, v_cache, block_tables = generate_caches_and_block_tables_vllm(
        k_unpad, v_unpad, kv_seq_lengths, BATCH_SIZE, MAX_NUM_BLOCKS_PER_SEQ, BLOCK_SIZE, dtype, device
    )

    block_tables = block_tables.to(device=device)
    max_seq_len_across_batch = kv_seq_lengths.max().item()
    output = torch.empty((BATCH_SIZE, NUM_ATTN_HEADS, HEAD_SIZE), dtype=dtype, device=device)
    sm_scale = 1.0 / (HEAD_SIZE**0.5)

    k_torch = convert_kv_unpad_to_padded(k_unpad, kv_seq_lengths, BATCH_SIZE, max_seq_len_across_batch)
    v_torch = convert_kv_unpad_to_padded(v_unpad, kv_seq_lengths, BATCH_SIZE, max_seq_len_across_batch)
    torch_padding_mask = create_attention_mask(kv_seq_lengths, BATCH_SIZE, q_len, max_seq_len_across_batch, device)

    if dtype == torch.float16:
        rtol = 1e-3
        atol = 1e-3

        high_precision_q = q.to(torch.float32)
        high_precision_k_torch = k_torch.to(torch.float32)
        high_precision_v_torch = v_torch.to(torch.float32)
        out_ref = torch_attn_ref(
            high_precision_q,
            high_precision_k_torch,
            high_precision_v_torch,
            torch_padding_mask,
            BATCH_SIZE,
            q_len,
            max_seq_len_across_batch,
            NUM_ATTN_HEADS,
            NUM_KV_HEADS,
            HEAD_SIZE,
        ).to(torch.float16)

    else:
        rtol = 1e-5
        atol = 1e-7

        out_ref = torch_attn_ref(
            q,
            k_torch,
            v_torch,
            torch_padding_mask,
            BATCH_SIZE,
            q_len,
            max_seq_len_across_batch,
            NUM_ATTN_HEADS,
            NUM_KV_HEADS,
            HEAD_SIZE,
        )

    alibi_slopes = None

    vllm_ops.paged_attention_v1(
        output,
        q.squeeze(2),
        k_cache,
        v_cache,
        NUM_KV_HEADS,
        sm_scale,
        block_tables,
        kv_seq_lengths,
        BLOCK_SIZE,
        max_seq_len_across_batch,
        alibi_slopes,
        "auto",
    )
    numpy_allclose(out_ref, output, rtol=rtol, atol=atol)


if __name__ == "__main__":
    BATCH_SIZE = [1, 4, 7, 32]
    BLOCK_SIZE = [8, 16, 32]
    MAX_NUM_BLOCKS_PER_SEQ = [1, 8, 32]
    HEAD_SIZE = [64, 128]
    NUM_ATTN_HEADS = [16]
    KV_GROUP_NUM = [1, 2, 16]
    DTYPE = [torch.float16, torch.float32]
    test_combinations = list(
        product(BATCH_SIZE, BLOCK_SIZE, MAX_NUM_BLOCKS_PER_SEQ, HEAD_SIZE, NUM_ATTN_HEADS, KV_GROUP_NUM, DTYPE)
    )
    for (
        batch_size,
        block_size,
        max_num_blocks_per_seq,
        head_size,
        num_attn_heads,
        kv_group_num,
        dtype,
    ) in test_combinations:
        test_flash_decoding_attention(
            batch_size, block_size, max_num_blocks_per_seq, head_size, num_attn_heads, kv_group_num, dtype
        )

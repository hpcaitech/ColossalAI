import pytest
import torch
from packaging import version

from colossalai.kernel.triton import flash_decoding_attention
from colossalai.utils import get_current_device
from tests.test_infer.test_ops.triton.kernel_utils import (
    convert_kv_unpad_to_padded,
    generate_caches_and_block_tables_v2,
    prepare_padding_mask,
    torch_attn_ref,
)

try:
    import triton  # noqa

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")

HEAD_DIM = 128


def prepare_data(
    bsz: int,
    num_attn_heads: int,
    num_kv_heads: int,
    head_dim: int,
    same_context_len: bool,
    q_len: int,
    max_kv_seq_len: int,
    dtype=torch.float16,
    device="cuda",
):
    # Use the provided maximum sequence length for each sequence when testing with teh same context length,
    # otherwise generate random context lengths.
    # returns
    #   q [bsz, num_attn_heads, q_len, head_dim]
    #   k_unpad/v_unpad [num_tokens, num_kv_heads, head_dim]
    kv_lengths = (
        torch.tensor([max_kv_seq_len for _ in range(bsz)], dtype=torch.int32, device=device)
        if same_context_len
        else torch.randint(low=1, high=max_kv_seq_len, size=(bsz,), dtype=torch.int32, device=device)
    )
    num_tokens = torch.sum(kv_lengths).item()

    q_size = (bsz, q_len, num_attn_heads, head_dim)
    q = torch.empty(size=q_size, dtype=dtype, device=device).normal_(mean=0.0, std=0.5).transpose(1, 2)
    kv_size = (num_tokens, 2 * num_kv_heads, head_dim)
    kv_unpad = torch.empty(size=kv_size, dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    k_unpad, v_unpad = torch.split(kv_unpad, [num_kv_heads, num_kv_heads], dim=-2)

    return q, k_unpad, v_unpad, kv_lengths


@pytest.mark.skipif(not (HAS_TRITON and TRITON_CUDA_SUPPORT), reason="requires triton")
@pytest.mark.parametrize("bsz", [4, 7, 32])
@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize("max_num_blocks_per_seq", [8, 32])
@pytest.mark.parametrize("num_attn_heads", [16])
@pytest.mark.parametrize("kv_group_num", [1, 2, 16])
@pytest.mark.parametrize("same_context_len", [True, False])
@pytest.mark.parametrize("q_len", [1, 5])
def test_flash_decoding(
    bsz: int,
    block_size: int,
    max_num_blocks_per_seq: int,
    num_attn_heads: int,
    kv_group_num: int,
    same_context_len: bool,
    q_len: int,
):
    torch.manual_seed(123)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    num_kv_heads = num_attn_heads // kv_group_num
    assert isinstance(num_kv_heads, int) and num_kv_heads > 0, "Invalid number of kv heads."
    max_seq_len = block_size * max_num_blocks_per_seq
    dtype = torch.float16
    device = get_current_device()
    q, k_unpad, v_unpad, kv_lengths = prepare_data(
        bsz, num_attn_heads, num_kv_heads, HEAD_DIM, same_context_len, q_len, max_seq_len, dtype, device
    )
    # The maximum sequence length in the batch (if context lengths randomly generated)
    max_kv_len_in_b = kv_lengths.max().item()

    k_torch = convert_kv_unpad_to_padded(k_unpad, kv_lengths, bsz, max_kv_len_in_b)
    v_torch = convert_kv_unpad_to_padded(v_unpad, kv_lengths, bsz, max_kv_len_in_b)
    torch_padding_mask = prepare_padding_mask(kv_lengths, bsz, q_len, max_kv_len_in_b, q.device)
    out_torch = torch_attn_ref(
        q, k_torch, v_torch, torch_padding_mask, bsz, q_len, max_kv_len_in_b, num_attn_heads, num_kv_heads, HEAD_DIM
    )

    k_cache, v_cache, block_tables = generate_caches_and_block_tables_v2(
        k_unpad, v_unpad, kv_lengths, bsz, max_num_blocks_per_seq, block_size, dtype, device
    )
    block_tables = block_tables.to(device=device)
    # The maximum block length splitted on kv should be the kv cache block size
    kv_max_split_num = (max_kv_len_in_b + block_size - 1) // block_size
    output = torch.empty((bsz * q_len, num_attn_heads, HEAD_DIM), dtype=q.dtype, device=q.device)
    mid_output = torch.empty(
        size=(bsz * q_len, num_attn_heads, kv_max_split_num, HEAD_DIM), dtype=torch.float32, device=q.device
    )
    mid_output_lse = torch.empty(
        size=(bsz * q_len, num_attn_heads, kv_max_split_num), dtype=torch.float32, device=q.device
    )
    sm_scale = 1.0 / (HEAD_DIM**0.5)
    # Here we use different methods to hide the q_len dimension,
    # refer to attention forward function in modeling.
    if q_len > 1:
        q = q.transpose(1, 2).contiguous()  # [bsz, q_len, num_heads, head_dim]
        q = q.view(-1, q.size(-2), q.size(-1))  # [bsz * q_len, num_heads, head_dim]
    else:
        q = q.squeeze(2)
    assert q.shape == (bsz * q_len, num_attn_heads, HEAD_DIM)

    out_triton = flash_decoding_attention(
        q,
        k_cache,
        v_cache,
        kv_lengths,
        block_tables,
        block_size,
        max_kv_len_in_b,
        output,
        mid_output,
        mid_output_lse,
        sm_scale=sm_scale,
        kv_group_num=kv_group_num,
        q_len=q_len,
    )  # [bsz * q_len, num_heads, head_dim]

    assert out_torch.shape == out_triton.shape
    assert torch.allclose(out_torch, out_triton, atol=1e-3, rtol=1e-4)


if __name__ == "__main__":
    test_flash_decoding(16, 32, 32, 16, 1, True)

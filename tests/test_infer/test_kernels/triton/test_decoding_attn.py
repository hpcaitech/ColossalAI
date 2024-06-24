import numpy as np
import pytest
import torch
from packaging import version

from colossalai.inference.utils import get_alibi_slopes
from colossalai.kernel.triton import flash_decoding_attention
from colossalai.utils import get_current_device
from tests.test_infer.test_kernels.triton.kernel_utils import (
    convert_kv_unpad_to_padded,
    create_attention_mask,
    generate_caches_and_block_tables_v2,
    generate_caches_and_block_tables_v3,
    torch_attn_ref,
)
from tests.test_infer.test_kernels.triton.test_context_attn_unpad import generate_alibi_mask

try:
    import triton  # noqa

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")

HEAD_DIM = 128


def numpy_allclose(x, y, rtol, atol):
    x_numpy = x.detach().cpu().numpy()
    y_numpy = y.detach().cpu().numpy()

    np.testing.assert_allclose(x_numpy, y_numpy, rtol=rtol, atol=atol)


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
@pytest.mark.parametrize("bsz", [7, 16])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("max_num_blocks_per_seq", [8, 16])
@pytest.mark.parametrize("num_attn_heads", [16])
@pytest.mark.parametrize("kv_group_num", [1, 4])
@pytest.mark.parametrize("same_context_len", [True, False])
@pytest.mark.parametrize("q_len", [1, 5])
@pytest.mark.parametrize("use_alibi_slopes", [True, False])
@pytest.mark.parametrize("use_new_kcache_layout", [True, False])
def test_flash_decoding(
    bsz: int,
    block_size: int,
    max_num_blocks_per_seq: int,
    num_attn_heads: int,
    kv_group_num: int,
    same_context_len: bool,
    q_len: int,
    use_alibi_slopes: bool,
    use_new_kcache_layout: bool,
):
    if use_new_kcache_layout and use_alibi_slopes:
        # TODO(yuanheng-zhao): Since the alibi kernel is pretty similar to the original one,
        # the code (alibi kernel) will be refactored later to avoid code duplication, when
        # the whole triton flow with new k cache layout has been supported and tested.
        # And tests for the alibi kernel using new kcache layout will be added then.
        pytest.skip("Alibi kernel does not support new kcache layout yet.")

    torch.manual_seed(123)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    num_kv_heads = num_attn_heads // kv_group_num
    assert isinstance(num_kv_heads, int) and num_kv_heads > 0, "Invalid number of kv heads."
    max_seq_len = block_size * max_num_blocks_per_seq
    dtype = torch.float32
    device = get_current_device()

    if use_alibi_slopes:
        alibi_slopes = get_alibi_slopes(num_attn_heads, device)
        # Currently, alibi flash decoding does not support q_len>1.
        q_len = 1
    else:
        alibi_slopes = None

    q, k_unpad, v_unpad, kv_lengths = prepare_data(
        bsz, num_attn_heads, num_kv_heads, HEAD_DIM, same_context_len, q_len, max_seq_len, dtype, device
    )
    # The maximum sequence length in the batch (if context lengths randomly generated)
    max_kv_len_in_b = kv_lengths.max().item()

    k_torch = convert_kv_unpad_to_padded(k_unpad, kv_lengths, bsz, max_kv_len_in_b)
    v_torch = convert_kv_unpad_to_padded(v_unpad, kv_lengths, bsz, max_kv_len_in_b)
    attention_mask = create_attention_mask(kv_lengths, bsz, q_len, max_kv_len_in_b, q.device)

    if use_alibi_slopes:
        alibi_mask = generate_alibi_mask(alibi_slopes, num_attn_heads, max_kv_len_in_b, q.device)
        attention_mask = attention_mask + alibi_mask

        if q_len == 1:
            if len(attention_mask.size()) == 4:
                attention_mask = attention_mask[:, :, -1:, :]
            else:
                attention_mask = attention_mask[:, -1:, :]

    out_torch = torch_attn_ref(
        q, k_torch, v_torch, attention_mask, bsz, q_len, max_kv_len_in_b, num_attn_heads, num_kv_heads, HEAD_DIM
    )

    if use_new_kcache_layout:
        k_cache, v_cache, block_tables = generate_caches_and_block_tables_v3(
            k_unpad, v_unpad, kv_lengths, bsz, max_num_blocks_per_seq, block_size, dtype, device
        )
    else:
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
        alibi_slopes=alibi_slopes,
        sm_scale=sm_scale,
        kv_group_num=kv_group_num,
        q_len=q_len,
        use_new_kcache_layout=use_new_kcache_layout,
    )  # [bsz * q_len, num_heads, head_dim]

    assert out_torch.shape == out_triton.shape

    rtol = 1e-4
    # After the shape becomes larger, some data elements are too small, leading to excessively large relative errors.
    if use_alibi_slopes:
        rtol = 100

    numpy_allclose(out_torch, out_triton, atol=1e-3, rtol=rtol)


if __name__ == "__main__":
    test_flash_decoding(16, 32, 32, 16, 1, True, 1, use_alibi_slopes=False, use_new_kcache_layout=True)

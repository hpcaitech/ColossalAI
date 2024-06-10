import pytest
import torch
from packaging import version

from colossalai.inference.utils import get_alibi_slopes
from colossalai.kernel.triton import context_attention_unpadded
from colossalai.utils import get_current_device
from tests.test_infer.test_kernels.triton.kernel_utils import (
    generate_caches_and_block_tables_v2,
    generate_caches_and_block_tables_v3,
    torch_attn_ref,
)

try:
    import triton  # noqa

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")

HEAD_DIM = 32


def _fill_with_neg_inf(t):
    return t.float().fill_(float("-inf")).type_as(t)


# alibi mask calculation adapted from https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat/blob/main/modeling_baichuan.py
def generate_alibi_mask(slopes, num_heads, max_seq_len, device):
    token_position = torch.arange(max_seq_len, device=device) - max_seq_len + 1
    token_position = token_position.unsqueeze(0).unsqueeze(0).expand(num_heads, -1, -1)
    diag = torch.diag(token_position[0])
    token_position = token_position - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * token_position
    alibi = alibi.view(num_heads, 1, max_seq_len)
    alibi_mask = torch.triu(_fill_with_neg_inf(torch.zeros([max_seq_len, max_seq_len], device=device)), 1)
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    return alibi_mask


def torch_attn_unpad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    context_lengths: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    slopes: torch.Tensor = None,
):
    # Process sequence one by one and concatenate them together.
    # q,k,v [num_tokens(sum(context_lengths)), num_heads, head_dim]
    assert context_lengths.dim() == 1, "context_lengths should be a 1D tensor"

    _, num_heads, head_dim = q.shape
    out_torch = []
    start_idx = 0
    for seq_i in range(len(context_lengths)):
        end_idx = start_idx + context_lengths[seq_i].item()
        seq_len = end_idx - start_idx
        mask = torch.tril(torch.ones(1, 1, seq_len, seq_len), diagonal=0).to(device=q.device)
        mask[mask == 0.0] = float("-inf")

        if slopes is not None:
            alibi_mask = generate_alibi_mask(slopes, num_heads, seq_len, q.device)
            mask = mask + alibi_mask

        torch_attn_ref_out = torch_attn_ref(
            q[start_idx:end_idx].unsqueeze(0).transpose(1, 2),
            k[start_idx:end_idx].unsqueeze(0).transpose(1, 2),
            v[start_idx:end_idx].unsqueeze(0).transpose(1, 2),
            mask,
            1,  # set bsz as 1 as we're processing sequence one by one
            seq_len,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
        )
        out_torch.append(torch_attn_ref_out.squeeze(0))
        start_idx = end_idx

    return torch.cat(out_torch, dim=0)


@pytest.mark.skipif(not (HAS_TRITON and TRITON_CUDA_SUPPORT), reason="requires triton")
@pytest.mark.parametrize("bsz", [7, 32])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("max_num_blocks_per_seq", [8, 16])
@pytest.mark.parametrize("num_attn_heads", [16])
@pytest.mark.parametrize("kv_group_num", [1, 4])
@pytest.mark.parametrize("same_context_len", [True, False])
@pytest.mark.parametrize("use_alibi_slopes", [True, False])
@pytest.mark.parametrize("use_new_kcache_layout", [True, False])
def test_context_attention(
    bsz: int,
    block_size: int,
    max_num_blocks_per_seq: int,
    num_attn_heads: int,
    kv_group_num: int,
    same_context_len: bool,
    use_alibi_slopes: bool,
    use_new_kcache_layout: bool,
):
    if use_new_kcache_layout and use_alibi_slopes:
        # TODO(yuanheng-zhao): Since the alibi kernel is pretty similar to the original one,
        # the code (alibi kernel) will be refactored later to avoid code duplication, when
        # the whole triton flow with new k cache layout has been supported and tested.
        # And tests for the alibi kernel using new kcache layout will be added then.
        return

    torch.manual_seed(123)
    # It's necessary to clear cache here.
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    num_kv_heads = num_attn_heads // kv_group_num
    assert isinstance(num_kv_heads, int) and num_kv_heads > 0, "Invalid number of kv heads."
    max_seq_len = max_num_blocks_per_seq * block_size
    dtype = torch.float16
    device = get_current_device()
    alibi_slopes = None

    if use_alibi_slopes:
        alibi_slopes = get_alibi_slopes(num_attn_heads, device)

    if same_context_len:
        context_lengths = torch.tensor([max_seq_len for _ in range(bsz)], dtype=torch.int32, device=device)
    else:
        context_lengths = torch.randint(low=1, high=max_seq_len, size=(bsz,), dtype=torch.int32, device=device)
    num_tokens = torch.sum(context_lengths).item()

    qkv_size = (num_tokens, num_attn_heads + 2 * num_kv_heads, HEAD_DIM)
    qkv_unpad = torch.empty(size=qkv_size, dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    q_unpad, k_unpad, v_unpad = torch.split(qkv_unpad, [num_attn_heads, num_kv_heads, num_kv_heads], dim=-2)
    q_unpad = q_unpad.contiguous()

    if use_new_kcache_layout:
        k_cache_ref, v_cache_ref, block_tables = generate_caches_and_block_tables_v3(
            k_unpad, v_unpad, context_lengths, bsz, max_num_blocks_per_seq, block_size, dtype, device
        )
    else:
        k_cache_ref, v_cache_ref, block_tables = generate_caches_and_block_tables_v2(
            k_unpad, v_unpad, context_lengths, bsz, max_num_blocks_per_seq, block_size, dtype, device
        )

    block_tables = block_tables.to(device=device)
    k_cache_triton = torch.zeros_like(k_cache_ref)
    v_cache_triton = torch.zeros_like(v_cache_ref)

    _, num_heads, head_dim = q_unpad.shape

    out_triton = context_attention_unpadded(
        q_unpad,
        k_unpad,
        v_unpad,
        k_cache_triton,
        v_cache_triton,
        context_lengths,
        block_tables,
        block_size,
        alibi_slopes=alibi_slopes,
        use_new_kcache_layout=use_new_kcache_layout,
    )

    out_triton = out_triton.view(-1, num_heads, head_dim)
    out_torch = torch_attn_unpad(q_unpad, k_unpad, v_unpad, context_lengths, num_attn_heads, num_kv_heads, alibi_slopes)

    assert out_torch.shape == out_triton.shape
    assert torch.allclose(out_torch, out_triton, atol=1e-3)
    assert torch.equal(k_cache_ref, k_cache_triton)
    assert torch.equal(v_cache_ref, v_cache_triton)


if __name__ == "__main__":
    test_context_attention(4, 32, 8, 16, 1, True, True, True)

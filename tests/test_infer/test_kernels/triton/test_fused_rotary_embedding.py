import pytest
import torch
from packaging import version

from colossalai.kernel.triton.fused_rotary_embedding import fused_rotary_embedding

try:
    import triton  # noqa

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")


def torch_rotary_emb(x, cos, sin):
    dim = x.shape[-1]
    x0 = x[:, :, : dim // 2]
    x1 = x[:, :, dim // 2 :]
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return torch.cat((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)


@pytest.mark.skipif(not (HAS_TRITON and TRITON_CUDA_SUPPORT), reason="requires triton")
@pytest.mark.parametrize("lengths", [[3, 4, 6, 7], [5, 1, 17]])
@pytest.mark.parametrize("num_heads, num_kv_heads", [(32, 32), (32, 8)])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("fused_qkv", [False, True])
def test_fused_rotary_emb(lengths, num_heads, num_kv_heads, head_dim, fused_qkv):
    torch.manual_seed(123)
    dtype = torch.float32
    device = "cuda"
    num_tokens = sum(lengths)

    if fused_qkv:
        # q and k are non-contiguous views of a packed qkv tensor, so their token strides differ
        # from their own shapes (and from each other when num_kv_heads < num_heads).
        qkv = torch.randn(num_tokens, num_heads + 2 * num_kv_heads, head_dim, dtype=dtype, device=device)
        q = qkv[:, :num_heads]
        k = qkv[:, num_heads : num_heads + num_kv_heads]
    else:
        q = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    q_ref, k_ref = q.clone(), k.clone()

    cos_cache = torch.randn(1024, head_dim, dtype=dtype, device=device)
    sin_cache = torch.randn(1024, head_dim, dtype=dtype, device=device)
    lengths = torch.tensor(lengths, dtype=torch.int32, device=device)

    # each token is rotated by its position within its own sequence
    positions = torch.cat([torch.arange(n, device=device) for n in lengths.tolist()])
    cos = cos_cache[positions, : head_dim // 2]
    sin = sin_cache[positions, : head_dim // 2]
    q_ref = torch_rotary_emb(q_ref, cos, sin)
    k_ref = torch_rotary_emb(k_ref, cos, sin)

    if fused_qkv:
        v_before = qkv[:, num_heads + num_kv_heads :].clone()

    fused_rotary_embedding(q, k, cos_cache, sin_cache, lengths)

    torch.testing.assert_close(q, q_ref)
    torch.testing.assert_close(k, k_ref)
    if fused_qkv:
        # rotating k must not spill into the neighbouring v heads
        torch.testing.assert_close(qkv[:, num_heads + num_kv_heads :], v_before)


if __name__ == "__main__":
    test_fused_rotary_emb([3, 4, 6, 7], 32, 8, 64, True)

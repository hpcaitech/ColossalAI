from copy import deepcopy

import pytest
import torch
from packaging import version

from colossalai.kernel.triton.fused_rotary_embedding import fused_rotary_embedding
from colossalai.kernel.triton.no_pad_rotary_embedding import rotary_embedding
from colossalai.kernel.triton.rotary_cache_copy import get_xine_cache

try:
    import triton  # noqa

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")


@pytest.mark.skip(reason="cuda error")
@pytest.mark.skipif(not (HAS_TRITON and TRITON_CUDA_SUPPORT), reason="requires triton")
def test_fused_rotary_emb():
    num_tokens = 20
    num_kv_heads = 32
    head_dim = 64
    dtype = torch.float32
    q_shape = (num_tokens, num_kv_heads, head_dim)
    q = -2.3 + 0.5 * torch.randn(q_shape, dtype=dtype, device="cuda")
    q_copy = deepcopy(q)

    k_shape = (num_tokens, num_kv_heads, head_dim)
    k = -2.3 + 0.5 * torch.randn(k_shape, dtype=dtype, device="cuda")
    k_copy = deepcopy(k)

    cos_shape = (1024, head_dim)
    lengths = torch.tensor([3, 4, 6, 7], device="cuda")
    cos_cache = -1.2 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    sin_cache = -2.0 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")

    cos, sin = get_xine_cache(lengths, cos_cache[:, : head_dim // 2], sin_cache[:, : head_dim // 2])

    rotary_embedding(q, k, cos, sin)
    fused_rotary_embedding(q_copy, k_copy, cos_cache, sin_cache, lengths)
    torch.allclose(q, q_copy)
    torch.allclose(k, k_copy)


if __name__ == "__main__":
    test_fused_rotary_emb()

import pytest
import torch
from packaging import version

from colossalai.kernel.triton import get_xine_cache

try:
    import triton  # noqa

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")


@torch.no_grad()
def get_cos_sin(lengths, cos_cache, sin_cache, is_prompts, dtype):
    """
    Get cos and sin for the cache, and return nopad format.
    Args:
        lengths: shape(num_seqs,), stores lenghth of each sequence.
        cos_cache: shape(max_rotary_position(e.g.2048), head_dim), cos cache constrcuted in model.
        sin_cache: shape(max_rotary_position(e.g.2048), head_dim), sin cache constrcuted in model.
        is_prompts: bool, mark if in prefill mode.
        dtype: The data type of this inference process.
    """

    if is_prompts:
        index_arrays = [torch.arange(length) for length in lengths]
    else:
        index_arrays = [(length - 1).view(-1) for length in lengths]
    indices = torch.cat(index_arrays, dim=-1)
    cos_output = cos_cache[indices].to(dtype=dtype)
    sin_output = sin_cache[indices].to(dtype=dtype)

    return (cos_output, sin_output)


@pytest.mark.skipif(
    not TRITON_CUDA_SUPPORT or not HAS_TRITON, reason="triton requires cuda version to be higher than 11.4"
)
@pytest.mark.parametrize("BATCH_SIZE", [4])
@pytest.mark.parametrize("MAX_SEQ_LEN", [64])
@pytest.mark.parametrize("HEAD_DIM", [64])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_get_xine_cache(BATCH_SIZE, MAX_SEQ_LEN, HEAD_DIM, dtype):
    MAX_TOTAL_TOKENS = BATCH_SIZE * MAX_SEQ_LEN
    cos_cache = torch.randn((MAX_TOTAL_TOKENS, HEAD_DIM), dtype=dtype, device="cuda")
    sin_cache = torch.randn((MAX_TOTAL_TOKENS, HEAD_DIM), dtype=dtype, device="cuda")
    lengths = torch.randint(2, MAX_SEQ_LEN, (BATCH_SIZE,), device="cuda")
    # prefill
    cos_ref, sin_ref = get_cos_sin(lengths, cos_cache, sin_cache, is_prompts=True, dtype=dtype)
    cos, sin = get_xine_cache(lengths, cos_cache, sin_cache, is_prompts=True)
    assert torch.allclose(cos, cos_ref)
    assert torch.allclose(sin, sin_ref)
    # decoding
    ncos_ref, nsin_ref = get_cos_sin(lengths, cos_cache, sin_cache, is_prompts=False, dtype=dtype)
    cos, sin = get_xine_cache(lengths, cos_cache, sin_cache, is_prompts=False)
    assert torch.allclose(cos, ncos_ref)
    assert torch.allclose(sin, nsin_ref)


if __name__ == "__main__":
    test_get_xine_cache(4, 64, 256, torch.float32)

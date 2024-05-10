import random

import pytest
import torch

from colossalai.kernel.kernel_loader import InferenceOpsLoader
from colossalai.utils import get_current_device

inference_ops = InferenceOpsLoader().load()

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [42]  # Arbitrary values for testing
NUM_LAYERS = [1]  # Arbitrary values for testing
NUM_HEADS = [8]  # Arbitrary values for testing
HEAD_SIZES = [64, 80, 96, 112, 128, 256]
BLOCK_SIZES = [8, 16, 32]


@pytest.mark.skipif(True, reason="FP8 conversion still needs improvement, now we skip it's relative test!")
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_size", [64, 80, 96, 112, 128, 256])
@pytest.mark.parametrize("block_size", [8, 16, 32])
@pytest.mark.parametrize("num_blocks", [1024, 10000])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16, torch.float])
@pytest.mark.parametrize("seed", [0])
@torch.inference_mode()
def test_fp8_conversion(
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = get_current_device()

    low = -224.0
    high = 224.0
    shape = (num_blocks, num_heads, head_size, block_size)
    cache = torch.empty(shape, dtype=dtype, device=device)
    cache.uniform_(low, high)

    cache_fp8 = torch.empty_like(cache, dtype=torch.uint8)
    inference_ops.convert_fp8(cache, cache_fp8)

    converted_cache = torch.empty_like(cache)
    inference_ops.convert_fp8(cache_fp8, converted_cache)

    assert torch.allclose(cache, converted_cache, atol=0.001, rtol=0.1)


if __name__ == "__main__":
    test_fp8_conversion(8, 64, 8, 1024, torch.half, 0)

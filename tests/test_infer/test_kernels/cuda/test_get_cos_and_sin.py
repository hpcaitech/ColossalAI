import numpy as np
import pytest
import torch

from colossalai.kernel.kernel_loader import InferenceOpsLoader
from tests.test_infer.test_kernels.triton.test_xine_copy import get_cos_sin

inference_ops = InferenceOpsLoader().load()


def numpy_equal(x, y):
    x_numpy = x.detach().cpu().numpy()
    y_numpy = y.detach().cpu().numpy()

    np.testing.assert_equal(x_numpy, y_numpy)


@pytest.mark.parametrize("BATCH_SIZE", [4])
@pytest.mark.parametrize("MAX_SEQ_LEN", [64])
@pytest.mark.parametrize("HEAD_DIM", [64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_get_cos_and_sin(BATCH_SIZE, MAX_SEQ_LEN, HEAD_DIM, dtype):
    MAX_TOTAL_TOKENS = BATCH_SIZE * MAX_SEQ_LEN
    cos_cache = torch.randn((MAX_TOTAL_TOKENS, HEAD_DIM), dtype=dtype, device="cuda")
    sin_cache = torch.randn((MAX_TOTAL_TOKENS, HEAD_DIM), dtype=dtype, device="cuda")
    lengths = torch.randint(2, MAX_SEQ_LEN, (BATCH_SIZE,), device="cuda").to(torch.int32)

    max_seq_len_in_batch = lengths.max()

    # prefill
    cos_ref, sin_ref = get_cos_sin(lengths, cos_cache, sin_cache, is_prompts=True, dtype=dtype)

    cos = torch.zeros_like(cos_ref)
    sin = torch.zeros_like(sin_ref)

    inference_ops.get_cos_and_sin(cos_cache, sin_cache, cos, sin, lengths, max_seq_len_in_batch, True)

    numpy_equal(cos, cos_ref)
    numpy_equal(sin, sin_ref)

    # decoding
    ncos_ref, nsin_ref = get_cos_sin(lengths, cos_cache, sin_cache, is_prompts=False, dtype=dtype)

    cos = torch.zeros_like(ncos_ref)
    sin = torch.zeros_like(nsin_ref)

    inference_ops.get_cos_and_sin(cos_cache, sin_cache, cos, sin, lengths, max_seq_len_in_batch, False)
    numpy_equal(cos, ncos_ref)
    numpy_equal(sin, nsin_ref)


if __name__ == "__main__":
    test_get_cos_and_sin(16, 4096, 256, torch.float16)

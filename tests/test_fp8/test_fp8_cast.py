import torch
from torch.testing import assert_close

from colossalai.accelerator import get_accelerator
from colossalai.quantization.fp8 import cast_from_fp8, cast_from_fp8_pipeline, cast_to_fp8, cast_to_fp8_pipeline
from colossalai.testing import parameterize


@parameterize("shape", [(100, 10), (10, 100), (3, 7), (2, 1), (1, 2), (2, 2), (4, 2), (5,), (4,), (2,)])
@parameterize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@parameterize("fp8_format", ["e4m3", "e5m2"])
def test_fp8_cast(shape, dtype, fp8_format):
    x = torch.rand(shape, dtype=dtype, device=get_accelerator().get_current_device())
    ret, scale_inv = cast_to_fp8(x, fp8_format=fp8_format)
    out = cast_from_fp8(ret, scale_inv, x.dtype)
    assert_close(out, x, rtol=0.1, atol=0.1)

    if x.size(-1) % 2 == 0:
        inp_dict = {"hidden_states": x.clone()}
        cast_to_fp8_pipeline(inp_dict)
        cast_from_fp8_pipeline(inp_dict)
        assert_close(inp_dict["hidden_states"], x, rtol=0.1, atol=0.1)


if __name__ == "__main__":
    test_fp8_cast()
